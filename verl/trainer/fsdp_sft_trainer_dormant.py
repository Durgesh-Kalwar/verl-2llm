# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

# Add vLLM imports
try:
    from vllm import SamplingParams
    from verl.third_party.vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Falling back to standard generation.")

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import CPUOffloadPolicy, MixedPrecisionPolicy, apply_fsdp2, fsdp2_clip_grad_norm_, fsdp2_load_full_state_dict, get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()
        
        # Initialize vLLM engine for validation generation if available
        self.vllm_engine = None
        if VLLM_AVAILABLE and self.device_mesh.get_rank() == 0:
            self._init_vllm_engine()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _init_vllm_engine(self):
        """Initialize vLLM engine for batch generation during validation"""
        try:
            # Initialize vLLM engine with the model directly (following vllm_rollout.py approach)
            tensor_parallel_size = min(2, torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
            max_model_len = getattr(self.config.data, 'max_length', 2048)
            
            self.vllm_engine = LLM(
                actor_module=self.fsdp_model,  # Pass the model directly instead of path
                tokenizer=self.tokenizer,
                model_hf_config=self.model_config,
                tensor_parallel_size=tensor_parallel_size,
                dtype=torch.bfloat16,  # Match the model dtype
                enforce_eager=True,  # Disable CUDA graphs for validation
                gpu_memory_utilization=0.8,
                skip_tokenizer_init=False,
                max_model_len=max_model_len,
                load_format="auto",
                disable_log_stats=True,
                max_num_batched_tokens=8192,
                enable_chunked_prefill=False,
            )
            
            # Offload vllm model to reduce peak memory usage
            self.vllm_engine.offload_model_weights()
            
            if self.device_mesh.get_rank() == 0:
                print("vLLM engine initialized successfully for validation generation")
                
        except Exception as e:
            print(f"Failed to initialize vLLM engine: {e}")
            self.vllm_engine = None

    def _update_vllm_engine(self):
        """Update vLLM engine with current model weights"""
        if self.vllm_engine is None or self.device_mesh.get_rank() != 0:
            return
            
        try:
            # Clean up existing engine
            if hasattr(self, 'vllm_engine') and self.vllm_engine is not None:
                self.vllm_engine.offload_model_weights()
                del self.vllm_engine
                torch.cuda.empty_cache()
            
            # Reinitialize vLLM engine with updated model (following vllm_rollout.py approach)
            tensor_parallel_size = min(2, torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
            max_model_len = getattr(self.config.data, 'max_length', 2048)
            
            self.vllm_engine = LLM(
                actor_module=self.fsdp_model,  # Pass the updated model directly
                tokenizer=self.tokenizer,
                model_hf_config=self.model_config,
                tensor_parallel_size=tensor_parallel_size,
                dtype=torch.bfloat16,  # Match the model dtype
                enforce_eager=True,
                gpu_memory_utilization=0.8,
                skip_tokenizer_init=False,
                max_model_len=max_model_len,
                load_format="auto",
                disable_log_stats=True,
                max_num_batched_tokens=8192,
                enable_chunked_prefill=False,
            )
            
            # Offload vllm model to reduce peak memory usage
            self.vllm_engine.offload_model_weights()
            
            if self.device_mesh.get_rank() == 0:
                print("vLLM engine updated with current model weights")
                
        except Exception as e:
            print(f"Failed to update vLLM engine: {e}")
            self.vllm_engine = None

    def _cleanup_vllm_engine(self):
        """Clean up vLLM engine"""
        if self.device_mesh.get_rank() == 0 and self.vllm_engine is not None:
            try:
                # Offload model weights before cleanup
                self.vllm_engine.offload_model_weights()
                del self.vllm_engine
                self.vllm_engine = None
                torch.cuda.empty_cache()
                        
                print("vLLM engine cleaned up successfully")
                
            except Exception as e:
                print(f"Error cleaning up vLLM engine: {e}")

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(self.model_config.max_position_embeddings, self.config.data.max_length)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")
    
    def _generate_and_evaluate_responses(self, batch, input_ids, loss_mask):
        """Generate responses during validation and decode them for evaluation using vLLM for batch processing"""
        # Only generate on rank 0 to avoid redundant computation
        if self.device_mesh.get_rank() != 0:
            return
            
        batch_size = input_ids.shape[0]
        
        # Find where questions end (where loss_mask starts being 1)
        # loss_mask has 0s for question tokens and 1s for response tokens
        loss_mask_2d = loss_mask.view(batch_size, -1)
        
        # Find the first position where loss_mask becomes 1 (start of response)
        question_lengths = []
        for i in range(batch_size):
            mask_row = loss_mask_2d[i]
            response_start = torch.argmax((mask_row == 1).int())
            if mask_row[response_start] == 1:
                question_lengths.append(response_start.item() + 1)  # +1 to include the token before response
            else:
                # If no response tokens found, use the full sequence
                question_lengths.append(mask_row.shape[0])
        
        # Extract ground truth responses first
        ground_truth_responses = []
        for i in range(batch_size):
            full_sequence = input_ids[i]
            response_mask = loss_mask_2d[i] == 1
            if response_mask.any():
                response_start_idx = torch.argmax(response_mask.int()).item()
                # Find end of response (last non-pad token)
                response_end_idx = len(full_sequence) - 1
                while response_end_idx > response_start_idx and full_sequence[response_end_idx] == self.tokenizer.pad_token_id:
                    response_end_idx -= 1
                
                ground_truth_ids = full_sequence[response_start_idx:response_end_idx + 1]
                ground_truth_text = self.tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
                ground_truth_responses.append(ground_truth_text)
            else:
                ground_truth_responses.append("")
        
        # Generate responses using vLLM if available, otherwise fall back to standard generation
        if self.vllm_engine is not None:
            generated_responses = self._generate_with_vllm(input_ids, question_lengths)
        else:
            generated_responses = self._generate_with_standard_model(input_ids, question_lengths)

        # Print some examples for debugging/monitoring
        print("\n" + "="*50)
        print("VALIDATION GENERATION SAMPLES:")
        print("="*50)
        for i in range(min(3, batch_size)):  # Show first 3 examples
            question_ids = input_ids[i, :question_lengths[i]]
            question_text = self.tokenizer.decode(question_ids, skip_special_tokens=True)
            
            print(f"\nSample {i+1}:")
            print(f"Question: {question_text}")
            print(f"Generated: {generated_responses[i]}")
            print(f"Ground Truth: {ground_truth_responses[i]}")
            print("-" * 30)
        print("="*50 + "\n")
        
        return generated_responses, ground_truth_responses

    def _generate_with_vllm(self, input_ids, question_lengths):
        """Generate responses using vLLM for efficient batch processing"""
        batch_size = input_ids.shape[0]
        generated_responses = []
        
        # Prepare prompt token ids for vLLM (following vllm_rollout.py approach)
        prompt_token_ids = []
        for i in range(batch_size):
            question_len = question_lengths[i]
            question_ids = input_ids[i, :question_len]
            # Remove left padding if any (similar to _pre_process_inputs in vllm_rollout.py)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            non_pad_indices = torch.nonzero(question_ids != pad_token_id, as_tuple=False)
            if len(non_pad_indices) > 0:
                start_idx = non_pad_indices[0][0]
                clean_question_ids = question_ids[start_idx:]
            else:
                clean_question_ids = question_ids
            prompt_token_ids.append(clean_question_ids.tolist())
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            min_p=0.90,
            max_tokens=1024,
            stop=None,  # Let the model decide when to stop
        )
        
        try:
            # Generate responses in batch using vLLM (following vllm_rollout.py approach)
            outputs = self.vllm_engine.generate(
                prompts=None,  # Use prompt_token_ids instead of text prompts
                sampling_params=sampling_params,
                prompt_token_ids=prompt_token_ids,
                use_tqdm=False,
            )
            
            # Extract generated text from outputs
            for output in outputs:
                generated_text = output.outputs[0].text
                generated_responses.append(generated_text)
                
        except Exception as e:
            print(f"Error in vLLM generation: {e}")
            # Fall back to standard generation
            generated_responses = self._generate_with_standard_model(input_ids, question_lengths)
        
        return generated_responses

    def _generate_with_standard_model(self, input_ids, question_lengths):
        """Fallback method using standard model generation"""
        batch_size = input_ids.shape[0]
        generated_responses = []
        
        for i in range(batch_size):
            question_len = question_lengths[i]
            
            # Extract question tokens (up to where response starts)
            question_ids = input_ids[i, :question_len].unsqueeze(0)
            
            # Create attention mask for question
            attention_mask = torch.ones_like(question_ids)
            
            try:
                with torch.no_grad():
                    generated_ids = self.fsdp_model.generate(
                        input_ids=question_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        temperature=0.8,
                        min_p=0.90,
                    )
                
                # Extract only the generated part (remove input)
                generated_response_ids = generated_ids[0, question_len:]
                generated_text = self.tokenizer.decode(generated_response_ids, skip_special_tokens=True)
                generated_responses.append(generated_text)
                
            except Exception as e:
                print(f"Error generating response for sample {i}: {e}")
                generated_responses.append("")
        
        return generated_responses

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            else:
                # During validation, generate responses to evaluate model performance
                print("Generating text sequences for test dataset")
                self._generate_and_evaluate_responses(batch, input_ids, loss_mask)
            
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        return {"train/loss": step_loss.detach().item(), "train/lr(1e-3)": lr * 1e3}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        
        # Update vLLM engine with current model weights before validation generation
        if self.device_mesh.get_rank() == 0 and VLLM_AVAILABLE:
            self._update_vllm_engine()
        
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.
        from typing import List
        def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
            # remove the left padding in the prompt token_id
            # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
            non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
            token_ids = prompt_token_ids[non_pad_index:].tolist()
            return token_ids

        eval_flag = True
        if eval_flag:
            for val_data in self.val_dataloader:
                val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(self.device_name)
                self.fsdp_model.eval()
                with torch.no_grad():
                    input_ids = val_data["val_input_ids"]
                    attention_mask = val_data["val_attention_mask"]
                    eos_token_id = self.tokenizer.eos_token_id
                    batch_size = input_ids.shape[0]

                    idx_list = []
                    for i in range(batch_size):
                        idx_list.append(_pre_process_inputs(self.tokenizer.pad_token_id, input_ids[i]))
                        
                    output = self.fsdp_model.generate(
                        input_ids=idx_list,
                        max_new_tokens=1024,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        temperature=0.8,
                        top_p=0.90,
                        use_cache=True
                    )
                    for i, seq in enumerate(output):
                        print(f"Sample {i + 1}: {self.tokenizer.decode(seq, skip_special_tokens=True)}")

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(self.train_dataloader, total=self.steps_per_epoch, desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}", disable=rank != 0):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    print("----validation started------")
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(self.device_name)
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    # Clean up vLLM engine
                    self._cleanup_vllm_engine()
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    print("processing validation dataset")
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer, val_flag=True)

    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    # try:
    trainer.fit()
    # finally:
    #     # Ensure cleanup happens even if there's an exception
    #     trainer._cleanup_vllm_engine()
    #     destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, val_flag=False):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    if val_flag:
        dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, val_flag=val_flag)
    else:
        dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()

import os
import torch
from torch.optim import AdamW
from peft import LoraConfig
from model_pipeline import HookedModelPipeline
from tuning.dataset_builder import build_training_dataset
from tuning.losses import consistency_kl_loss

def run_lora_training():
    print("🚀 Initializing Phase C LoRA Training Loop & Checkpoint Saving...")
    
    # 8. Attach LoRA (Targeting c_attn and c_proj as specified in Phase C)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Initialize pipeline with peft configuration (base weights are frozen automatically by PEFT)
    pipeline = HookedModelPipeline(model_key="gpt2", peft_config=peft_config)    
    # 9. Write the training loop
    optimizer = AdamW(filter(lambda p: p.requires_grad, pipeline.model.parameters()), lr=1e-4)
    training_samples = build_training_dataset()
    
    pipeline.model.train()
    epochs = 2
    alpha = 1.0
    beta = 0.5
    
    step_count = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for sample in training_samples:
            optimizer.zero_grad()
            
            prompt = sample["baseline_prompt"]
            pert_prompt = sample["perturbed_prompt"]
            target_text = sample["ground_truth_answer"]
            truth_score = sample["truth_score"]
            
            # Forward pass: baseline prompt + target
            loss_base, logits_base = pipeline.forward_with_grad(prompt, target_text)
            
            # Forward pass: perturbed prompt for KL consistency loss
            _, logits_pert = pipeline.forward_with_grad(pert_prompt, target_text)
            
            # Compute auxiliary terms
            kl_val = consistency_kl_loss(logits_base, logits_pert)
            advantage = (truth_score - 75.0) / 100.0
            reward_val = advantage * loss_base
            
            # Combined multi-objective loss
            combined_loss = loss_base + (alpha * kl_val) + (beta * reward_val)
            
            # Backward pass & Optimizer step
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            step_count += 1
            
            if step_count % 5 == 0:
                print(f"  [Step {step_count}] Running Loss: {combined_loss.item():.4f} | Current Truth-Score Signal: {truth_score}")
                
        print(f"✅ Epoch {epoch+1}/{epochs} completed | Total Epoch Loss: {total_loss:.4f}")
        
    # 10. Save the checkpoint natively using PEFT format (stores only the small adapter weights)
    checkpoint_dir = "checkpoints/lora_v1"
    os.makedirs(checkpoint_dir, exist_ok=True)
    pipeline.model.save_pretrained(checkpoint_dir)
    print(f"🎯 PEFT-native LoRA adapter successfully saved to {checkpoint_dir}/")

if __name__ == "__main__":
    run_lora_training()
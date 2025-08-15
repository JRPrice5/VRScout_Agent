"""
PPO Training Monitor for ACT RL Training

This script provides comprehensive monitoring of PPO training metrics
by integrating the diagnostic tools to track key metrics during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import time
import os

from .debug_log_prob import debug_log_prob_computation, analyze_ppo_ratio, suggest_log_prob_fixes
from .ppo_diagnostics import diagnose_ppo_training_issues, check_data_quality, suggest_hyperparameter_adjustments, emergency_training_fixes


class PPOTrainingMonitor:
    """
    Monitors PPO training metrics and provides diagnostics.
    """
    
    def __init__(
        self,
        log_dir: str = "logs/ppo_monitor",
        save_plots: bool = True,
        log_frequency: int = 1,
        alert_threshold: Dict[str, float] = None
    ):
        """
        Initialize the PPO training monitor.
        
        Args:
            log_dir: Directory to save logs and plots
            save_plots: Whether to save plots
            log_frequency: How often to log metrics (every N updates)
            alert_threshold: Thresholds for alerting on metrics
        """
        self.log_dir = log_dir
        self.save_plots = save_plots
        self.log_frequency = log_frequency
        
        # Create log directory if it doesn't exist
        if save_plots and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Default alert thresholds
        self.alert_threshold = alert_threshold or {
            "policy_loss": 10.0,
            "value_loss": 10.0,
            "kl_div": 0.1,
            "ratio_max": 2.0,
            "ratio_min": 0.5,
            "entropy": 0.01
        }
        
        # Initialize metrics tracking
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "kl_div": [],
            "entropy": [],
            "ratio_mean": [],
            "ratio_min": [],
            "ratio_max": [],
            "advantage_mean": [],
            "advantage_std": [],
            "return_mean": [],
            "return_std": [],
            "log_prob_diff_mean": [],
            "log_prob_diff_std": [],
            "update_count": 0,
            "extreme_ratio_count": 0,
            "nan_inf_count": 0,
            "timestamp": []
        }
    
    def monitor_update(
        self,
        policy_loss: float,
        value_loss: float,
        kl_div: float,
        entropy: float,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        epoch: int = 0,
        batch: int = 0
    ):
        """
        Monitor a single PPO update step.
        """
        self.metrics["update_count"] += 1
        self.metrics["timestamp"].append(time.time())
        
        # Record basic metrics
        self.metrics["policy_loss"].append(policy_loss)
        self.metrics["value_loss"].append(value_loss)
        self.metrics["kl_div"].append(kl_div)
        self.metrics["entropy"].append(entropy)
        
        # Compute and record ratio metrics
        log_prob_diff = new_log_probs - old_log_probs
        ratio = torch.exp(torch.clamp(log_prob_diff, min=-20.0, max=20.0))
        
        self.metrics["ratio_mean"].append(ratio.mean().item())
        self.metrics["ratio_min"].append(ratio.min().item())
        self.metrics["ratio_max"].append(ratio.max().item())
        
        self.metrics["log_prob_diff_mean"].append(log_prob_diff.mean().item())
        self.metrics["log_prob_diff_std"].append(log_prob_diff.std().item())
        
        # Record advantage and return statistics
        self.metrics["advantage_mean"].append(advantages.mean().item())
        self.metrics["advantage_std"].append(advantages.std().item())
        self.metrics["return_mean"].append(returns.mean().item())
        self.metrics["return_std"].append(returns.std().item())
        
        # Check for extreme values
        if (ratio > self.alert_threshold["ratio_max"]).any() or (ratio < self.alert_threshold["ratio_min"]).any():
            self.metrics["extreme_ratio_count"] += 1
            print(f"\n[ALERT] Extreme PPO ratios detected in update {self.metrics['update_count']}, epoch {epoch}, batch {batch}")
            analyze_ppo_ratio(old_log_probs, new_log_probs, name=f"Update_{self.metrics['update_count']}")
        
        # Check for NaN/Inf values
        if (torch.isnan(ratio).any() or torch.isinf(ratio).any() or
            torch.isnan(advantages).any() or torch.isinf(advantages).any() or
            torch.isnan(returns).any() or torch.isinf(returns).any()):
            self.metrics["nan_inf_count"] += 1
            print(f"\n[ALERT] NaN or Inf values detected in update {self.metrics['update_count']}, epoch {epoch}, batch {batch}")
        
        # Run diagnostics at specified frequency
        if self.metrics["update_count"] % self.log_frequency == 0:
            self._run_diagnostics(
                policy_loss=policy_loss,
                value_loss=value_loss,
                kl_div=kl_div,
                entropy=entropy,
                advantages=advantages,
                returns=returns,
                rewards=rewards,
                actions=actions,
                mu=mu,
                logvar=logvar
            )
            
            if self.save_plots:
                self._save_plots()
    
    def _run_diagnostics(
        self,
        policy_loss: float,
        value_loss: float,
        kl_div: float,
        entropy: float,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None
    ):
        """
        Run diagnostic tools on current metrics.
        """
        print(f"\n=== PPO MONITOR: UPDATE {self.metrics['update_count']} ===")
        
        # Run PPO diagnostics
        issues, suggestions = diagnose_ppo_training_issues(
            policy_loss=policy_loss,
            value_loss=value_loss,
            kl_div=kl_div,
            entropy=entropy
        )
        
        # Check data quality
        if rewards is not None:
            check_data_quality(rewards, advantages, returns)
        else:
            check_data_quality(torch.zeros_like(advantages), advantages, returns)
        
        # Debug log probability computation if mu and logvar are available
        if actions is not None and mu is not None and logvar is not None:
            debug_log_prob_computation(
                actions=actions,
                mu=mu,
                logvar=logvar,
                name=f"Update_{self.metrics['update_count']}"
            )
        
        # Suggest fixes if issues were found
        if issues:
            suggest_log_prob_fixes()
            suggest_hyperparameter_adjustments()
            
            # If severe issues persist, suggest emergency fixes
            if len(issues) >= 3 or self.metrics["extreme_ratio_count"] > 5:
                emergency_training_fixes()
        
        print(f"=== END PPO MONITOR ===\n")
    
    def _save_plots(self):
        """
        Save plots of training metrics.
        """
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot policy metrics
        axs[0, 0].plot(self.metrics["policy_loss"], label="Policy Loss")
        axs[0, 0].set_title("Policy Loss")
        axs[0, 0].set_xlabel("Update")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)
        
        # Plot value metrics
        axs[0, 1].plot(self.metrics["value_loss"], label="Value Loss")
        axs[0, 1].set_title("Value Loss")
        axs[0, 1].set_xlabel("Update")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].grid(True)
        
        # Plot KL divergence
        axs[1, 0].plot(self.metrics["kl_div"], label="KL Divergence")
        axs[1, 0].set_title("KL Divergence")
        axs[1, 0].set_xlabel("Update")
        axs[1, 0].set_ylabel("KL")
        axs[1, 0].grid(True)
        
        # Plot entropy
        axs[1, 1].plot(self.metrics["entropy"], label="Entropy")
        axs[1, 1].set_title("Entropy")
        axs[1, 1].set_xlabel("Update")
        axs[1, 1].set_ylabel("Entropy")
        axs[1, 1].grid(True)
        
        # Plot ratio statistics
        axs[2, 0].plot(self.metrics["ratio_mean"], label="Mean")
        axs[2, 0].plot(self.metrics["ratio_min"], label="Min")
        axs[2, 0].plot(self.metrics["ratio_max"], label="Max")
        axs[2, 0].set_title("PPO Ratio Statistics")
        axs[2, 0].set_xlabel("Update")
        axs[2, 0].set_ylabel("Ratio")
        axs[2, 0].legend()
        axs[2, 0].grid(True)
        
        # Plot advantage statistics
        axs[2, 1].plot(self.metrics["advantage_mean"], label="Mean")
        axs[2, 1].plot(self.metrics["advantage_std"], label="Std")
        axs[2, 1].set_title("Advantage Statistics")
        axs[2, 1].set_xlabel("Update")
        axs[2, 1].set_ylabel("Advantage")
        axs[2, 1].legend()
        axs[2, 1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/ppo_metrics_{self.metrics['update_count']}.png")
        plt.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training metrics.
        """
        if not self.metrics["policy_loss"]:
            return {"status": "No data collected yet"}
        
        return {
            "updates": self.metrics["update_count"],
            "policy_loss": {
                "mean": np.mean(self.metrics["policy_loss"]),
                "min": np.min(self.metrics["policy_loss"]),
                "max": np.max(self.metrics["policy_loss"]),
                "last": self.metrics["policy_loss"][-1]
            },
            "value_loss": {
                "mean": np.mean(self.metrics["value_loss"]),
                "min": np.min(self.metrics["value_loss"]),
                "max": np.max(self.metrics["value_loss"]),
                "last": self.metrics["value_loss"][-1]
            },
            "kl_div": {
                "mean": np.mean(self.metrics["kl_div"]),
                "min": np.min(self.metrics["kl_div"]),
                "max": np.max(self.metrics["kl_div"]),
                "last": self.metrics["kl_div"][-1]
            },
            "entropy": {
                "mean": np.mean(self.metrics["entropy"]),
                "min": np.min(self.metrics["entropy"]),
                "max": np.max(self.metrics["entropy"]),
                "last": self.metrics["entropy"][-1]
            },
            "ratio": {
                "mean": np.mean(self.metrics["ratio_mean"]),
                "min": np.min(self.metrics["ratio_min"]),
                "max": np.max(self.metrics["ratio_max"]),
                "last_mean": self.metrics["ratio_mean"][-1],
                "last_min": self.metrics["ratio_min"][-1],
                "last_max": self.metrics["ratio_max"][-1]
            },
            "extreme_ratio_count": self.metrics["extreme_ratio_count"],
            "nan_inf_count": self.metrics["nan_inf_count"]
        }


# Example usage
if __name__ == "__main__":
    # This is a demonstration of how to use the monitor
    monitor = PPOTrainingMonitor(log_dir="logs/ppo_demo")
    
    # Simulate some updates with random data
    for i in range(10):
        monitor.monitor_update(
            policy_loss=float(np.random.normal(0, 1)),
            value_loss=float(np.random.normal(0, 2)),
            kl_div=float(np.random.exponential(0.05)),
            entropy=float(np.random.normal(1, 0.2)),
            old_log_probs=torch.randn(64),
            new_log_probs=torch.randn(64),
            advantages=torch.randn(64),
            returns=torch.randn(64),
            rewards=torch.randn(64),
            epoch=i,
            batch=0
        )
    
    # Print summary
    print(monitor.get_summary())

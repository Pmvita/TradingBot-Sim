"""Agent registry for different RL algorithms."""

from typing import Dict, Type, Any
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy


class AgentRegistry:
    """Registry for RL agents and their configurations."""
    
    def __init__(self) -> None:
        """Initialize the registry."""
        self._agents: Dict[str, Type] = {
            "ppo": PPO,
            "dqn": DQN,
            "a2c": A2C,
        }
        
        self._default_policies: Dict[str, Type] = {
            "ppo": ActorCriticPolicy,
            "dqn": DQNPolicy,
            "a2c": ActorCriticPolicy,
        }
    
    def get_agent_class(self, algo: str) -> Type:
        """Get agent class by algorithm name.
        
        Args:
            algo: Algorithm name (ppo, dqn, a2c)
            
        Returns:
            Agent class
        """
        if algo not in self._agents:
            raise ValueError(f"Unknown algorithm: {algo}. Available: {list(self._agents.keys())}")
        
        return self._agents[algo]
    
    def get_policy_class(self, algo: str) -> Type:
        """Get default policy class for algorithm.
        
        Args:
            algo: Algorithm name
            
        Returns:
            Policy class
        """
        if algo not in self._default_policies:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        return self._default_policies[algo]
    
    def get_available_algorithms(self) -> list:
        """Get list of available algorithms.
        
        Returns:
            List of algorithm names
        """
        return list(self._agents.keys())
    
    def create_agent(
        self,
        algo: str,
        env,
        policy_kwargs: Dict[str, Any] = None,
        **kwargs
    ):
        """Create an agent instance.
        
        Args:
            algo: Algorithm name
            env: Environment instance
            policy_kwargs: Policy keyword arguments
            **kwargs: Additional agent arguments
            
        Returns:
            Agent instance
        """
        agent_class = self.get_agent_class(algo)
        policy_class = self.get_policy_class(algo)
        
        if policy_kwargs is None:
            policy_kwargs = {}
        
        return agent_class(
            policy=policy_class,
            env=env,
            policy_kwargs=policy_kwargs,
            **kwargs
        )


# Global registry instance
registry = AgentRegistry()

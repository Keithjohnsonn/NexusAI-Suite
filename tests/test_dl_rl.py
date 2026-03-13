import torch
import numpy as np
from src.vision.model import VisionModelFactory
from src.rl.dqn_agent import DQNAgent

def test_vision_model_creation():
    """
    Test creation of Vision models with custom head.
    """
    num_classes = 5
    model = VisionModelFactory.create_model("resnet50", num_classes=num_classes, pretrained=False)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (1, num_classes)

def test_dqn_agent_step():
    """
    Test DQN agent action selection and update logic.
    """
    state_dim = 4
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim, batch_size=2)
    
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    
    assert action in range(action_dim)
    
    # Store experience and check update
    agent.store_transition(state, action, 1.0, state, False)
    agent.store_transition(state, action, 1.0, state, False)
    
    loss = agent.update()
    assert loss is not None

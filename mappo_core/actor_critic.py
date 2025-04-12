import torch
import torch.nn as nn
import config

class ActorCritic(nn.Module):
    """ Actor-Critic model for multi-agent PPO with communication """

    def __init__(self, obs_shape):
        """ Initializes the model with CNN encoder, communication, and actor-critic networks """
        super().__init__()
        n_agents = config.NUM_ROBOTS
        hidden_dim = 64

        # CNN Encoder
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # State encoder
        conv_out_size = 32 * obs_shape[1] * obs_shape[2]
        self.enc_hidden_dim = hidden_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(conv_out_size, self.enc_hidden_dim),
            nn.ReLU()
        )

        # Distance embedding
        self.distance_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU()
        )

        # Communication module
        self.comm_net = nn.Sequential(
            nn.Linear(self.enc_hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )

        # Update actor network
        actor_input_size = hidden_dim + (hidden_dim * (n_agents-1))  # Adjusted for communication
        self.actor = nn.Sequential(
            nn.Linear(actor_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Force output to be 4 actions
        )

        # Update critic network
        critic_input_size = hidden_dim * n_agents + n_agents * 2
        self.critic = nn.Sequential(
            nn.Linear(critic_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode_and_communicate(self, obs_list, agent_positions):
        """ Encodes observations and performs inter-agent communication """
        encoded = []

        for i, obs in enumerate(obs_list):
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)

            # Get conv features
            conv_features = self.conv_encoder(obs)
            enc = self.encoder(conv_features)

            # Process position features
            pos = agent_positions[i]
            if len(pos.shape) == 1:
                pos = pos.unsqueeze(0)
            dist_features = self.distance_embedding(pos)

            # Combine features
            combined = torch.cat([enc, dist_features], dim=-1)
            encoded.append(combined)

        # Generate messages
        messages = []
        for enc in encoded:
            msg = self.comm_net(enc)
            messages.append(msg)

        # Combine messages for each agent
        features = []
        for i in range(len(encoded)):
            other_messages = messages[:i] + messages[i+1:]

            if other_messages:
                # Stack and reshape messages
                other_msgs = torch.cat(other_messages, dim=-1)
                combined = torch.cat([messages[i], other_msgs], dim=-1)
            else:
                # Pad for single agent case
                combined = torch.cat([messages[i], torch.zeros_like(messages[i])], dim=-1)

            features.append(combined)

        return features, messages

    def forward(self, obs_list, agent_positions):
        """ Forward pass to compute action logits and value estimates """
        obs_list = [obs.to(self.device) for obs in obs_list]
        agent_positions = [pos.to(self.device) for pos in agent_positions]

        features, messages = self.encode_and_communicate(obs_list, agent_positions)

        # print("Features shape:", [f.shape for f in features])

        # Get action logits for each agent
        action_logits = []
        for feat in features:
            logits = self.actor(feat)
            # Scale logits and normalize
            scaled_logits = logits / config.MIN_POLICY_STD
            # Center logits around mean to prevent extreme probabilities
            scaled_logits = scaled_logits - scaled_logits.mean(dim=-1, keepdim=True)
            action_logits.append(scaled_logits)

            # print("Raw logits:", logits)
            # print("Scaled and centered logits:", scaled_logits)
            # print("Action probabilities:", F.softmax(scaled_logits, dim=-1))

        try:
            batch_size = messages[0].size(0)
            all_messages = torch.cat([msg for msg in messages], dim=1)
            all_positions = torch.cat([pos.view(batch_size, -1) for pos in agent_positions], dim=1)
            joint_state = torch.cat([all_messages, all_positions], dim=1)
            value = self.critic(joint_state)

            # print("Value output:", value)

        except RuntimeError as e:
            print(f"Debug shapes - messages: {[m.shape for m in messages]}")
            print(f"Debug shapes - positions: {[p.shape for p in agent_positions]}")
            raise e

        return action_logits, value

    def get_value(self, obs_list, agent_positions):
        """ Computes the value function for given observations and positions """
        with torch.no_grad():
            obs_list = [obs.to(self.device) if not isinstance(obs, torch.Tensor) 
                       else obs.to(self.device) for obs in obs_list]
            agent_positions = [pos.to(self.device) if not isinstance(pos, torch.Tensor)
                             else pos.to(self.device) for pos in agent_positions]

            batch_size = obs_list[0].size(0) if len(obs_list[0].shape) > 3 else 1

            # Process observations
            encoded = []
            for obs in obs_list:
                if len(obs.shape) == 3:
                    obs = obs.unsqueeze(0)
                enc = self.encoder(self.conv_encoder(obs))
                encoded.append(enc)

            # Combine encoded states
            all_encoded = torch.cat(encoded, dim=1)
            all_positions = torch.cat([pos.view(batch_size, -1) for pos in agent_positions], dim=1)
            joint_state = torch.cat([all_encoded, all_positions], dim=1)

            return self.critic(joint_state)

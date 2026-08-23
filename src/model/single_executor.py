import torch
from torch import nn


class SingleExecutor(nn.Module):
    def __init__(self, embedding_dim: int = 16):
        super().__init__()

        self.value_embedding = nn.Embedding(10, embedding_dim)
        self.opcode_embedding = nn.Embedding(6, embedding_dim)

        # R0, R1, R2, R3, NONE
        self.register_embedding = nn.Embedding(5, embedding_dim)

        # 4 state values + opcode + arg1 + arg2
        input_dim = 7 * embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 40),
        )

    def forward(self, state, opcode, arg1, arg2):
        """
        state:  [B, 4]
        opcode: [B]
        arg1:   [B]
        arg2:   [B]

        returns:
            logits: [B, 4, 10]
        """

        # [B, 4] -> [B, 4, E]
        state_emb = self.value_embedding(state)

        # [B, 4, E] -> [B, 4E]
        state_emb = state_emb.flatten(start_dim=1)

        # [B] -> [B, E]
        opcode_emb = self.opcode_embedding(opcode)
        arg1_emb = self.register_embedding(arg1)
        arg2_emb = self.register_embedding(arg2)

        x = torch.cat(
            [state_emb, opcode_emb, arg1_emb, arg2_emb],
            dim=-1,
        )

        # [B, 40]
        logits = self.network(x)

        # [B, 4, 10]
        return logits.view(-1, 4, 10)
import torch
from torch import nn


class MultiStepExecutor(nn.Module):
    def __init__(
        self,
        embedding_dim=16,
        hidden_dim=128,
    ):
        super().__init__()

        self.value_embedding = nn.Embedding(
            10,
            embedding_dim
        )

        self.opcode_embedding = nn.Embedding(
            6,
            embedding_dim
        )

        # R0-R3 + NONE
        self.register_embedding = nn.Embedding(
            5,
            embedding_dim
        )

        self.encoder = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.GELU(),
        )

        instruction_dim = 3 * embedding_dim

        self.step_network = nn.Sequential(
            nn.Linear(
                hidden_dim + instruction_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.decoder = nn.Linear(
            hidden_dim,
            4 * 10,
        )

    def encode_state(self, state):
        # state: [B, 4]

        state_emb = self.value_embedding(state)

        # [B, 4, E] -> [B, 4E]
        state_emb = state_emb.flatten(start_dim=1)

        # [B, 128]
        hidden = self.encoder(state_emb)

        return hidden

    def encode_instruction(
        self,
        opcode,
        arg1,
        arg2,
    ):
        opcode_emb = self.opcode_embedding(opcode)
        arg1_emb = self.register_embedding(arg1)
        arg2_emb = self.register_embedding(arg2)

        return torch.cat(
            [
                opcode_emb,
                arg1_emb,
                arg2_emb,
            ],
            dim=-1,
        )

    def step(
        self,
        hidden,
        opcode,
        arg1,
        arg2,
    ):
        instruction_emb = self.encode_instruction(
            opcode,
            arg1,
            arg2,
        )

        x = torch.cat(
            [hidden, instruction_emb],
            dim=-1,
        )

        delta = self.step_network(x)

        hidden = hidden + delta

        return hidden

    def decode_state(self, hidden):
        logits = self.decoder(hidden)

        return logits.view(
            -1,
            4,
            10,
        )

    def forward(
        self,
        initial_state,
        opcodes,
        arg1s,
        arg2s,
    ):
        hidden = self.encode_state(initial_state)

        program_length = opcodes.size(1)

        for t in range(program_length):
            hidden = self.step(
                hidden,
                opcodes[:, t],
                arg1s[:, t],
                arg2s[:, t],
            )

        logits = self.decode_state(hidden)

        return logits
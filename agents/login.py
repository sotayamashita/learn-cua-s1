"""Harbor agent for the main.py text sample."""

import asyncio
import json
import os
from time import perf_counter

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

CUA_MODEL = "cua-ai/cua-s1-4b-0.2"


class LoginAgent(BaseAgent):
    """Score the shared login sample on the host; Docker only grades the choice."""

    def __init__(self, *args, device: str = "mps", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    @staticmethod
    def name() -> str:
        return "login-sample"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        started = perf_counter()
        if self.model_name == CUA_MODEL:
            await asyncio.to_thread(self._load_cua)
        elif not os.environ.get("TYPESAFE_API_KEY"):
            raise ValueError("Set TYPESAFE_API_KEY in .env before running Jev.")
        self.setup_seconds = perf_counter() - started

    def _load_cua(self) -> None:
        from main import check_environment, create_model

        dtype = check_environment(self.device)
        self.model = create_model("text", self.device, dtype)
        self.model.load()

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Save input and probabilities, then upload the prediction for grading.

        Harbor calls setup first. Loading time is recorded separately; inference
        time includes one cold inference or one API request, including network.
        The fixture contains model inputs only, never the verifier's answer.
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.logs_dir / "input.json"
        await environment.download_file("/app/login.json", input_path)
        sample = json.loads(input_path.read_text())
        if self.model_name == CUA_MODEL:
            from main import score_actions

            started = perf_counter()
            scores = await asyncio.to_thread(score_actions, self.model, sample)
            elapsed = perf_counter() - started
            probabilities = {item.element_id: item.probability for item in scores}
            resolved_model = CUA_MODEL
        else:
            from typesafe_sdk import AsyncTypeSafeClient, Choice

            async with AsyncTypeSafeClient(
                model=self.model_name, timeout=120
            ) as client:
                started = perf_counter()
                response = await client.system_one(
                    state=sample,
                    questions={
                        "action": Choice(
                            instructions=(
                                "Select the next action that accomplishes `goal` "
                                "given the current screen in `ax_tree`."
                            ),
                            criteria={
                                option["element_id"]: option
                                for option in sample["options"]
                            },
                        )
                    },
                )
                elapsed = perf_counter() - started
            probabilities = response.choices["action"].probabilities
            resolved_model = response.model
            context.n_input_tokens = response.usage.input_tokens
            context.n_output_tokens = response.usage.output_tokens
            (self.logs_dir / "response.json").write_text(
                response.model_dump_json(indent=2) + "\n"
            )

        candidate_ids = [option["element_id"] for option in sample["options"]]
        if set(probabilities) != set(candidate_ids) or any(
            not 0 <= p <= 1 for p in probabilities.values()
        ):
            raise ValueError("Model returned invalid candidate probabilities")
        prediction = {
            "model": resolved_model,
            "device": self.device if self.model_name == CUA_MODEL else "remote API",
            "modality": "text",
            "selected": max(candidate_ids, key=probabilities.__getitem__),
            "probabilities": probabilities,
            "setup_seconds": self.setup_seconds,
            "inference_seconds": elapsed,
        }
        path = self.logs_dir / "prediction.json"
        path.write_text(json.dumps(prediction, indent=2) + "\n")
        await environment.upload_file(path, "/app/prediction.json")

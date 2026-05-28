import os
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
<<<<<<< HEAD
from sglang.test.run_eval import run_eval
=======
from sglang.test.ascend.test_ascend_utils import write_results_to_github_step_summary
>>>>>>> 505f37a63dbcf376ee122592295d027bfa2e6094
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestVLMModels(CustomTestCase):
    model = ""
    mmmu_accuracy = 0.00
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.35,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        4,
    ]
    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    @classmethod
    def setUpClass(cls):
        # Removed argument parsing from here
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.base_url}/v1"

<<<<<<< HEAD
    def _run_vlm_mmmu_test(self, test_name="", custom_env=None):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        process = None
=======
        os.environ["TRANSFORMERS_VERBOSITY"] = os.getenv(
            "TRANSFORMERS_VERBOSITY", "error"
        )

    def run_mmmu_eval(
        self,
        model_version: str,
        output_path: str,
        limit: str,
        *,
        env: dict | None = None,
    ):
        """
        Evaluate a VLM on the MMMU validation set with lmms‑eval.
        Only `model_version` (checkpoint) and `chat_template` vary;
        We are focusing only on the validation set due to resource constraints.
        """
        # -------- fixed settings --------
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 2
        log_suffix = "openai_compatible"
        os.makedirs(output_path, exist_ok=True)

        # -------- compose --model_args --------
        model_args = f'model_version="{model_version}",' f"tp={tp}"

        # -------- build command list --------
        cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            model,
            "--model_args",
            model_args,
            "--tasks",
            tasks,
            "--batch_size",
            str(batch_size),
            "--log_samples",
            "--log_samples_suffix",
            log_suffix,
            "--output_path",
            str(output_path),
            "--limit",
            limit,
            "--config",
            "/__w/sglang/sglang/test/registered/ascend/vlm_models/mmmu-val.yaml",
        ]

        subprocess.run(
            cmd,
            check=True,
            timeout=3600,
        )

        return subprocess.list2cmdline(cmd)  # Return the command for logging purposes

    def _run_vlm_mmmu_test(
        self,
        output_path="./logs",
        test_name="",
        custom_env=None,
        capture_output=False,
        limit="50",
    ):
        """
        Common method to run VLM MMMU benchmark test.
        Args:
            model: Model to test
            output_path: Path for output logs
            test_name: Optional test name for logging
            custom_env: Optional custom environment variables
            capture_output: Whether to capture server stdout/stderr
        """
        print(f"\nTesting model: {self.model}{test_name}")

        model_metrics = {
            "server": subprocess.list2cmdline(map(str, self.other_args)),
            "client": "mmmu_eval",
            "accuracy_threshold": self.mmmu_accuracy,
        }

        process = None
        server_output = ""
        mmmu_accuracy = None
>>>>>>> 505f37a63dbcf376ee122592295d027bfa2e6094

        try:
            # Prepare environment variables
            process_env = os.environ.copy()
            if custom_env:
                process_env.update(custom_env)

            process = popen_launch_server(
                self.model,
                base_url=self.base_url,
                timeout=self.timeout_for_server_launch,
                api_key=self.api_key,
                other_args=self.other_args,
                env=process_env,
            )

<<<<<<< HEAD
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmmu",
                num_examples=100,
                num_threads=64,
                max_tokens=30,
            )
=======
            model_metrics["server"] = subprocess.list2cmdline(process.args)

            # Run evaluation
            model_metrics["client"] = self.run_mmmu_eval(self.model, output_path, limit)
>>>>>>> 505f37a63dbcf376ee122592295d027bfa2e6094

            args.return_latency = True

            metrics, latency = run_eval(args)

            metrics["score"] = round(metrics["score"], 4)
            metrics["latency"] = round(latency, 4)
            print(
                f"{'=' * 42}\n{self.model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
            )

<<<<<<< HEAD
=======
            # Capture server output if requested
            if capture_output and process:
                server_output = self._read_output_from_files()

            model_metrics["accuracy"] = mmmu_accuracy

            # Assert performance meets expected threshold
>>>>>>> 505f37a63dbcf376ee122592295d027bfa2e6094
            self.assertGreaterEqual(
                metrics["score"],
                self.mmmu_accuracy,
                f"Model {self.model} accuracy ({metrics['score']}) below expected threshold ({self.mmmu_accuracy:.4f}){test_name}",
            )

        except Exception as e:
            model_metrics["error"] = e
            print(f"Error testing {self.model}{test_name}: {e}")
            self.fail(f"Test failed for {self.model}{test_name}: {e}")
        finally:
            write_results_to_github_step_summary({self.model: model_metrics})

            # Ensure process cleanup happens regardless of success/failure
            if process is not None and process.poll() is None:
                print(f"Cleaning up process {process.pid}")
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")

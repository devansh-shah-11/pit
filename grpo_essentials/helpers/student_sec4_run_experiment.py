def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def compute_eval_loss(model, eval_prompts, eval_resps, tokenizer, device, max_batches=20):
    model.eval()

    total_logprob = 0.0
    total_tokens = 0.0

    with torch.no_grad():
        for i in range(min(max_batches, len(eval_prompts))):
            try:
                res = run_tokenize_prompt_and_output_util(
                    [eval_prompts[i]], [eval_resps[i]], tokenizer
                )
                input_ids = res['input_ids'].to(device)
                labels = res['labels'].to(device)
                response_mask = res['response_mask'].to(device)

                res_logprobs = run_get_response_log_probs_util(
                    model, input_ids, labels, return_token_entropy=False
                )
                log_probs = res_logprobs['log_probs']

                total_logprob += (log_probs * response_mask).sum().item()
                total_tokens += response_mask.sum().item()

            except Exception as e:
                print("Eval loss error", e)

    model.train()

    if total_tokens == 0:
        return 0

    return - total_logprob / total_tokens
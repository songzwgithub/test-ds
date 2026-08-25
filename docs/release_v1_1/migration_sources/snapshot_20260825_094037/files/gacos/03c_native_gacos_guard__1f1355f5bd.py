from pathlib import Path

p = Path("pystamps/cli.py")
text = p.read_text(encoding="utf-8")

old = '''    if str(getattr(run_config.runtime, "backend", "auto")) == "native":
        payload = _run_native_pipeline(args, run_config)
        print(json.dumps(payload, indent=2))
        return 1 if any(result["status"] == "failed" for result in payload) else 0
'''

new = '''    backend = str(
        getattr(
            run_config.runtime,
            "backend",
            "auto",
        )
    ).strip().lower()

    if (
        backend == "native"
        and bool(run_config.gacos.enabled)
        and args.end_step >= 7
        and args.start_step <= 8
    ):
        raise SystemExit(
            "Config error: GACOS correction for Stage 7/8 "
            "requires Python pipeline orchestration. "
            "Set runtime.backend to auto, threads, processes, "
            "or gpu."
        )

    if backend == "native":
        payload = _run_native_pipeline(
            args,
            run_config,
        )
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
        return (
            1
            if any(
                result["status"] == "failed"
                for result in payload
            )
            else 0
        )
'''

n = text.count(old)

if n != 1:
    raise RuntimeError(
        f"native dispatch block: expected 1 match, found {n}"
    )

text = text.replace(old, new, 1)

p.write_text(
    text.rstrip() + "\n",
    encoding="utf-8",
)

print("03c NATIVE GACOS GUARD: PASS")

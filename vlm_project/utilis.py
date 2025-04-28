def normalize_response(s: str) -> str:
    return s.strip().lower().replace("back", "behind").replace("forward", "front")
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import fire

# Color mapping for visualization (used for bounding boxes)
COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
}

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


def extract_kart_objects(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100):
    with open(info_path) as f:
        info = json.load(f)

    frame = info["detections"][view_index]
    karts = []

    for detection in frame:
        class_id, track_id, x1, y1, x2, y2 = detection
        if int(class_id) != 1:
            continue

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        kart_info = {
            "track_id": int(track_id),
            "center": (center_x, center_y),
            "kart_name": info["karts"][int(track_id)],
        }
        karts.append(kart_info)

    for k in karts:
        k["is_ego"] = k["track_id"] == 0

    return karts


def extract_track_info(info_path: str) -> str:
    with open(info_path) as f:
        info = json.load(f)
    return info["track"]


def generate_qa_pairs(info_path: str, view_index: int, subfolder: str, img_width: int = 150, img_height: int = 100):
    frame_id = Path(info_path).stem.replace("_info", "")
    image_file = f"{subfolder}/{frame_id}_{view_index:02d}_im.jpg"

    qa_pairs = []
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    track_name = extract_track_info(info_path)

    ego_karts = [k for k in karts if k["is_ego"]]
    if not ego_karts:
        return []

    ego = ego_karts[0]
    ego_center = ego["center"]

    qa_pairs.append({"question": "What kart is the ego car?", "answer": ego["kart_name"], "image_file": image_file})
    qa_pairs.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts)), "image_file": image_file})
    qa_pairs.append({"question": "What track is this?", "answer": track_name, "image_file": image_file})

    for kart in karts:
        if kart["is_ego"]:
            continue
        dx = kart["center"][0] - ego_center[0]
        dy = kart["center"][1] - ego_center[1]
        vert = "front" if dy < 0 else "behind"
        horiz = "left" if dx < 0 else "right"

        qa_pairs.append({
            "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
            "answer": vert,
            "image_file": image_file
        })

        qa_pairs.append({
            "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
            "answer": horiz,
            "image_file": image_file
        })

        qa_pairs.append({
            "question": f"Where is {kart['kart_name']} relative to the ego car?",
            "answer": f"{vert} and {horiz}",
            "image_file": image_file
        })

    front = sum(1 for k in karts if not k["is_ego"] and k["center"][1] < ego_center[1])
    back = sum(1 for k in karts if not k["is_ego"] and k["center"][1] > ego_center[1])
    left = sum(1 for k in karts if not k["is_ego"] and k["center"][0] < ego_center[0])
    right = sum(1 for k in karts if not k["is_ego"] and k["center"][0] > ego_center[0])

    qa_pairs.extend([
        {"question": "How many karts are in front of the ego car?", "answer": str(front), "image_file": image_file},
        {"question": "How many karts are behind the ego car?", "answer": str(back), "image_file": image_file},
        {"question": "How many karts are to the left of the ego car?", "answer": str(left), "image_file": image_file},
        {"question": "How many karts are to the right of the ego car?", "answer": str(right), "image_file": image_file},
    ])

    return qa_pairs


def generate_pairs_for_folder(subfolder: str, output_name: str):
    all_pairs = []
    data_dir = Path(f"D:/work/automation/work2/orginal4/data/{subfolder}")
    for info_file in data_dir.glob("*_info.json"):
        for view_index in range(9):
            try:
                pairs = generate_qa_pairs(str(info_file), view_index, subfolder)
                if not pairs:
                    print(f"⚠️ No QA pairs for: {info_file.name}, view {view_index}")
                else:
                    print(f"✅ {info_file.name} view {view_index}: {len(pairs)} QAs")
                    all_pairs.extend(pairs)
            except Exception as e:
                print(f"❌ Skipped {info_file.name} view {view_index}: {e}")

    output_file = data_dir / output_name
    with open(output_file, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\n✅ Generated {len(all_pairs)} QA pairs at {output_file}")


def main():
    fire.Fire({
        "generate_train": lambda: generate_pairs_for_folder("train", "balanced_qa_pairs.json"),
        "generate_valid": lambda: generate_pairs_for_folder("valid", "valid_qa_pairs.json"),
    })


if __name__ == "__main__":
    main()

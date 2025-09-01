def validate_yolo_annotation(file_path: str):
    """
    Validate the YOLO annotation file.
    Each line must be in the format:
    <object-class> <x_center> <y_center> <width> <height>
    """
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                return f"Invalid number of parts on line {i+1}: {len(parts)}"

            try:
                object_class = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                return f"Invalid data type on line {i+1}"

            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                return f"Values out of range on line {i+1}"

    return None

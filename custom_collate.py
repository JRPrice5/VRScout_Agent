import torch


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in the batch.

    Args:
        batch: List of tuples (image_data, qpos_data, action_data, is_pad, audio_data)

    Returns:
        Batched tensors with proper padding
    """
    # Separate the batch components
    image_data, qpos_data, action_data, is_pad, audio_data = zip(*batch)

    # Stack images - these should all be the same size
    image_data = torch.stack(image_data, dim=0)

    # Stack qpos - these should all be the same size
    qpos_data = torch.stack(qpos_data, dim=0)

    # Stack audio - these should all be the same size
    audio_data = torch.stack(audio_data, dim=0)

    # For action_data which has variable lengths, we need to pad
    # First determine the max length
    max_len = max([a.shape[0] for a in action_data])

    # Create padded action data
    padded_action_data = []
    for act in action_data:
        # If action is shorter than max_len, pad with zeros
        if act.shape[0] < max_len:
            padding = torch.zeros(
                (max_len - act.shape[0], act.shape[1]), dtype=act.dtype
            )
            padded_act = torch.cat([act, padding], dim=0)
        else:
            padded_act = act
        padded_action_data.append(padded_act)

    # Stack the padded actions
    action_data = torch.stack(padded_action_data, dim=0)

    # For is_pad, we also need to handle variable lengths
    padded_is_pad = []
    for pad in is_pad:
        if len(pad) < max_len:
            padding = torch.ones(max_len - len(pad), dtype=torch.bool)
            padded_pad = torch.cat([pad, padding], dim=0)
        else:
            padded_pad = pad
        padded_is_pad.append(padded_pad)

    # Stack the padded is_pad
    is_pad = torch.stack(padded_is_pad, dim=0)

    return image_data, qpos_data, action_data, is_pad, audio_data

import json
import uuid
from datetime import datetime
from enum import Enum


class MisinfoState(Enum):
    TRUE = 0
    FAKE = 1
    NOT_CHECKED = 2


class PostModel:
    def __init__(
        self,
        id: uuid.UUID,
        message: str,
        username: str,
        misinfo_state: MisinfoState,
        submitted_date: datetime,
    ):
        self.id = id
        self.message = message
        self.username = username
        self.misinfo_state = misinfo_state
        self.submitted_date = submitted_date


class MisinformationReport:
    def __init__(self, post_id: uuid.UUID, misinfo_state: MisinfoState):
        self.post_id = post_id
        self.misinfo_state = misinfo_state

    # MisinfoState is not serialisable, so we have to convert it into a dict
    # manually so json.dumps knows how to handle it
    def to_dict(self):
        match (self.misinfo_state):
            case MisinfoState.FAKE:
                state = 0
            case MisinfoState.TRUE:
                state = 1
            case MisinfoState.NOT_CHECKED:
                state = 2

        return {
            "post_id": str(self.post_id),
            "misinfo_state": state,
        }


def post_from_json(json_str: str) -> PostModel:
    data = json.loads(json_str)
    id = uuid.UUID(data["id"])
    message = data["message"]
    username = data["username"]
    submitted_date = datetime.fromisoformat(data["date"])

    misinfo_state_int = int(data["misinfo_state"])
    match (misinfo_state_int):
        case 0:
            misinfo_state = MisinfoState.FAKE
        case 1:
            misinfo_state = MisinfoState.TRUE
        case 2:
            misinfo_state = MisinfoState.NOT_CHECKED
        case _:
            raise ValueError(
                f"misinfo state can only be 0,1,2, got {misinfo_state_int}"
            )

    return PostModel(id, message, username, misinfo_state, submitted_date)

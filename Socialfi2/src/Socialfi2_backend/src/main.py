from kybra import query

message: str = ' '
@query
def greet(name: str) -> str:
    return f"Hello, {name}!"

def set_message(new_message: str) -> void

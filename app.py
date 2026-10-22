"""
Main entry point for the Face Lock AI application.

app.py is responsible only for orchestration and user interaction:
- show a terminal menu
- call registration workflow
- call live recognition workflow
- handle invalid input and top-level errors

AI logic, webcam logic, and embedding logic live in src/.
"""

from __future__ import annotations

from typing import Callable

from src.ai_engine import DEFAULT_MODEL_NAME, load_face_model
from src.operator_auth import verify_operator_pin
from src.register import register_user
from src.recognize import recognize_live
from src.utils import ensure_project_folders, print_status


# Menu choices are constants so they are easy to update later.
REGISTER_CHOICE = "1"
RECOGNIZE_CHOICE = "2"
EXIT_CHOICE = "3"


def print_divider() -> None:
    """
    Print a simple divider line for clean terminal formatting.
    """
    print("-" * 48)


def show_banner() -> None:
    """
    Show the application title.
    """
    print_divider()
    print("Face Lock AI System")
    print_divider()


def show_menu() -> None:
    """
    Show available user actions.
    """
    print()
    print("Choose an option:")
    print("1. Register new user")
    print("2. Start live recognition")
    print("3. Exit")
    print()


def get_menu_choice() -> str:
    """
    Read and clean the user's menu choice.

    Returns:
        The selected menu choice as a string.
    """
    # input() always returns text, so we strip spaces before comparing choices.
    return input("Enter choice: ").strip()


def pause_for_user() -> None:
    """
    Pause after an action so the user can read terminal messages.
    """
    input("\nPress Enter to continue...")


def run_safely(action_name: str, action: Callable[[], object]) -> object | None:
    """
    Run one application action with top-level exception handling.

    Args:
        action_name: Friendly name used in error messages.
        action: Function to run.

    Returns:
        The action result, or None if the action failed.
    """
    try:
        # Call the workflow function passed into this helper.
        return action()
    except KeyboardInterrupt:
        # Ctrl+C should stop the current action cleanly.
        print()
        print_status(f"{action_name} interrupted by user.", level="warning")
    except Exception as error:
        # app.py catches unexpected application-level errors at the boundary.
        print_status(f"{action_name} failed: {error}", level="error")

    return None


def handle_registration() -> None:
    """
    Start the registration workflow and print a short result summary.

    The operator must authenticate before registration begins.
    In demo mode (no PIN hash configured) a warning is logged and the
    operation is allowed through unconditionally.
    """
    print_status("Opening registration workflow.")

    # Operator authentication gate — protects enrollment from unauthorized use.
    # verify_operator_pin() handles demo mode, PIN prompting, and audit logging.
    if not verify_operator_pin():
        print_status(
            "Registration denied: operator authentication failed.",
            level="error",
        )
        return

    # register_user owns username input, webcam capture, and embedding saving.
    result = run_safely("Registration", register_user)

    if not isinstance(result, dict):
        return

    if result.get("success"):
        print_status(
            f"Registered {result['user_name']} with "
            f"{result['embedding_count']} embedding(s).",
            level="success",
        )
    else:
        reason = result.get("reason", "unknown_error")
        print_status(f"Registration was not completed: {reason}", level="warning")


def handle_recognition() -> None:
    """
    Start the live recognition workflow and print a short result summary.
    """
    print_status("Opening live recognition workflow.")

    # recognize_live owns webcam recognition and access decision display.
    result = run_safely("Live recognition", recognize_live)

    if not isinstance(result, dict):
        return

    if result.get("access_granted"):
        print_status(f"Last result: ACCESS GRANTED for {result['user_name']}.")
    else:
        reason = result.get("secondary_text", "access denied")
        print_status(f"Last result: ACCESS DENIED ({reason}).", level="warning")


def handle_menu_choice(choice: str) -> bool:
    """
    Run the workflow selected by the user.

    Args:
        choice: Cleaned menu choice.

    Returns:
        True if the app should keep running, False if it should exit.
    """
    if choice == REGISTER_CHOICE:
        handle_registration()
        pause_for_user()
        return True

    if choice == RECOGNIZE_CHOICE:
        handle_recognition()
        pause_for_user()
        return True

    if choice == EXIT_CHOICE:
        print_status("Exiting Face Lock AI System. Goodbye.", level="success")
        return False

    print_status("Invalid choice. Please enter 1, 2, or 3.", level="warning")
    pause_for_user()
    return True


def warmup_model(model_name: str = DEFAULT_MODEL_NAME) -> None:
    """
    Pre-load the DeepFace recognition model at startup.

    DeepFace downloads and caches model weights on first use. Loading the
    model here — before the user opens the menu — moves that one-time cost
    to startup rather than the first recognition or registration attempt,
    eliminating the visible freeze on first use.

    If the load fails (DeepFace not installed, network unavailable, or any
    other error) a warning is printed and startup continues normally.
    Recognition will still work; it will just pay the warm-up cost on first
    call instead.

    Args:
        model_name: The DeepFace model to pre-load (default: Facenet512).
    """
    print_status(f"Loading face recognition model ({model_name})…")

    try:
        load_face_model(model_name)
        print_status(f"Model ready: {model_name}.", level="success")
    except Exception as error:
        # A warm-up failure is non-fatal — the app can still operate.
        # The first recognition attempt will trigger the load again.
        print_status(
            f"Model pre-load skipped: {error}. "
            "Recognition will load the model on first use.",
            level="warning",
        )


def run_app() -> None:
    """
    Run the main application loop.
    """
    from src.integrity import enforce_strict_mode_policy
    enforce_strict_mode_policy(require_operator_auth=True)

    # Create required folders once when the app starts.
    ensure_project_folders()

    # Pre-load the face recognition model so the first recognition or
    # registration attempt does not pay the cold-start penalty.
    warmup_model()

    # Keep showing the menu until the user chooses Exit.
    keep_running = True

    while keep_running:
        show_banner()
        show_menu()

        # Read the user's menu selection.
        choice = get_menu_choice()

        # Run the selected workflow.
        keep_running = handle_menu_choice(choice)


def main() -> None:
    """
    Application entry point with final safety handling.
    """
    try:
        run_app()
    except KeyboardInterrupt:
        print()
        print_status("Application stopped by user.", level="warning")
    except Exception as error:
        print_status(f"Application error: {error}", level="error")


if __name__ == "__main__":
    # This keeps app.py import-safe and runs the app only when executed directly.
    main()

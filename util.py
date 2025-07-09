def pretty_print(text: str, subtext: str, color: str = "91") -> None:
    """
    Prints a formatted string to the console with colored and framed headers.

    Args:
        text (str): The main content to print.
        subtext (str): A label for the printed content such as AI/Human message.
        color (str): ANSI color code as a string.

    Returns:
        None
    """
    bar = "=" * 80
    spaced_subtext = " " + subtext + " "
    print(bar)
    print(f"\033[{color}m" + spaced_subtext.center(len(bar), "=") + "\033[0m")
    print(bar)
    if text!= "":
        print("\n" + text + "\n")
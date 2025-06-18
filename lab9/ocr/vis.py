"""Visualization module for OCR test results."""

import pandas as pd


def display_results(results):
    """Display OCR test results in a formatted table."""
    df = pd.DataFrame(results)

    print("\nFULL TEST RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\nSUMMARY")
    print("=" * 60)

    print("\nBest accuracy by test type:")
    best_by_test = df.loc[df.groupby("test_type")["accuracy"].idxmax()]
    print(
        best_by_test[
            [
                "test_type",
                "font",
                "noise_level",
                "rotation_angle",
                "confidence",
                "accuracy",
            ]
        ].to_string(index=False)
    )

    print("\nAverage accuracy by font:")
    font_avg = df.groupby("font")["accuracy"].mean().reset_index()
    font_avg["accuracy"] = font_avg["accuracy"].round(2)
    print(font_avg.to_string(index=False))

    print("\nAverage accuracy by confidence level:")
    conf_avg = df.groupby("confidence")["accuracy"].mean().reset_index()
    conf_avg["accuracy"] = conf_avg["accuracy"].round(2)
    print(conf_avg.to_string(index=False))

    print("\nNoise test results (accuracy %):")
    noise_pivot = (
        df[df["test_type"] == "noise"]
        .pivot_table(
            values="accuracy",
            index=["font", "noise_level"],
            columns="confidence",
            aggfunc="mean",
        )
        .round(2)
    )
    print(noise_pivot.to_string())

    print("\nRotation test results (accuracy %):")
    rotation_pivot = (
        df[df["test_type"] == "rotation"]
        .pivot_table(
            values="accuracy",
            index=["font", "rotation_angle"],
            columns="confidence",
            aggfunc="mean",
        )
        .round(2)
    )
    print(rotation_pivot.to_string())

from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg


def bind_application_font(size: int = 17) -> None:
    """Use the installed Microsoft YaHei font for complete Chinese coverage."""

    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        return
    with dpg.font_registry():
        font = dpg.add_font(str(font_path), size)
    dpg.bind_font(font)


def bind_application_theme() -> None:
    """Apply a restrained workstation theme with high-contrast status accents."""

    with dpg.theme() as theme, dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (16, 19, 21, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (22, 26, 28, 255))
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (24, 28, 30, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (59, 67, 70, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (31, 36, 38, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (43, 50, 52, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (52, 61, 63, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (20, 24, 26, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (25, 30, 32, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (39, 47, 49, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (54, 67, 68, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (67, 84, 82, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Header, (38, 47, 49, 255))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (51, 64, 65, 255))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (64, 79, 78, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Tab, (28, 33, 35, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (47, 58, 59, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TabActive, (52, 69, 67, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (226, 231, 230, 255))
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (123, 132, 132, 255))
        dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (93, 199, 164, 255))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (93, 199, 164, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Separator, (54, 62, 65, 255))
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 2)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 10)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 5)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7)
    dpg.bind_theme(theme)

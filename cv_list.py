import cv2

# 列出所有包含 "COLOR_" 的屬性
color_conversion_flags = [flag for flag in dir(cv2) if flag.startswith('COLOR_')]
print(color_conversion_flags)


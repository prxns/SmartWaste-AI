"""Conservative segregation guidance, separate from mathematical inference."""
GUIDANCE = {
    "cardboard": "Flatten cardboard and keep it dry. Remove food-soiled sections.",
    "glass": "Empty and rinse containers. Separate broken glass safely; check local collection rules.",
    "metal": "Empty metal packaging and remove residue. Do not puncture pressurized containers.",
    "paper": "Keep paper clean and dry. Coated and food-soiled paper may need separate collection.",
    "plastic": "Check the resin code and local acceptance list. Empty and rinse accepted packaging.",
    "trash": "Check for hazardous items and use the locally appropriate residual-waste stream.",
}


def guidance(category, status):
    text = GUIDANCE[category]
    if status == "Low":
        text += " Low suitability: clean or separate the item if practical before checking recycling eligibility."
    elif status == "Medium":
        text += " Moderate suitability: additional cleaning or sorting may be needed."
    else:
        text += " High suitability under the chosen assumptions; confirm local acceptance."
    return text

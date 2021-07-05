

def check_if_field_empty(field, field_placeholder):
    empty_fields = []

    # if not all_field_filled:  # IF there are blank fields, iterate and produce error message
    for key, value in field.items():
        if value == "":
            field_placeholder[key].error(
                f"Please do not leave field blank")
            empty_fields.append(key)

        else:
            pass

    return not empty_fields
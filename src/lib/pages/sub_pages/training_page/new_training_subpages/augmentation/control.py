import streamlit as st
from streamlit import session_state

# TODO: check the keys for each config

# set this to False to stop showing debugging values on sidebar
DEBUG = False


def select_num_interval(
    param_name: str, limits_list: list, defaults, n_for_hash, **kwargs
):
    # NOTE: the `n_for_hash` is the index from the `tranform_names`, check `get_transformations_params`
    st.sidebar.subheader(param_name)
    key = f"aug_{param_name}_{n_for_hash}"
    min_max_interval = st.sidebar.slider(
        "",
        limits_list[0],
        limits_list[1],
        defaults,
        key=key,
    )
    if DEBUG:
        st.sidebar.write(f"select_num_interval")
        st.sidebar.write(f"{key = }")
        st.sidebar.write(f"{session_state[key] = }")
    return min_max_interval


def select_several_nums(
    param_name, subparam_names, limits_list, defaults_list, n_for_hash, **kwargs
):
    st.sidebar.subheader(param_name)
    result = []
    assert len(limits_list) == len(defaults_list)
    assert len(subparam_names) == len(defaults_list)

    for name, limits, defaults in zip(subparam_names, limits_list, defaults_list):
        key = f"aug_{param_name}_{name}_{n_for_hash}"
        result.append(
            st.sidebar.slider(
                name,
                limits[0],
                limits[1],
                defaults,
                key=key,
            )
        )
        if DEBUG:
            st.sidebar.write(f"select_several_nums")
            st.sidebar.write(f"{key = }")
            st.sidebar.write(f"{session_state[key] = }")
    return tuple(result)


def select_min_max(
    param_name, limits_list, defaults_list, n_for_hash, min_diff=0, **kwargs
):
    assert len(param_name) == 2
    result = list(
        select_num_interval(
            " & ".join(param_name), limits_list, defaults_list, n_for_hash
        )
    )
    if result[1] - result[0] < min_diff:
        diff = min_diff - result[1] + result[0]
        if result[1] + diff <= limits_list[1]:
            result[1] = result[1] + diff
        elif result[0] - diff >= limits_list[0]:
            result[0] = result[0] - diff
        else:
            result = limits_list
    if DEBUG:
        st.sidebar.write(f"select_min_max")
    return tuple(result)


def select_RGB(param_name, n_for_hash, defaults_list=[0, 0, 0], **kwargs):
    result = select_several_nums(
        param_name,
        subparam_names=["Red", "Green", "Blue"],
        limits_list=[[0, 255], [0, 255], [0, 255]],
        defaults_list=defaults_list,
        n_for_hash=n_for_hash,
    )
    if DEBUG:
        st.sidebar.write(f"select_RGB")
    return tuple(result)


def replace_none(string):
    if string == "None":
        return None
    else:
        return string


def select_radio(param_name, options_list, n_for_hash, default_str=None, **kwargs):
    st.sidebar.subheader(param_name)
    key = f"aug_{param_name}{n_for_hash}"
    if default_str:
        # this is from the existing data that has been submitted before
        idx = options_list.index(default_str)
    else:
        idx = 0
    result = st.sidebar.radio(
        "", options_list, index=idx, key=key)
    if DEBUG:
        st.sidebar.write(f"select_radio")
        st.sidebar.write(f"{key = }")
        st.sidebar.write(f"{session_state[key] = }")
    return replace_none(result)


def select_checkbox(param_name, defaults, n_for_hash, **kwargs):
    st.sidebar.subheader(param_name)
    key = f"aug_{param_name}{n_for_hash}"
    result = st.sidebar.checkbox(
        "True", defaults, key=key
    )
    if DEBUG:
        st.sidebar.write(f"select_checkbox")
        st.sidebar.write(f"{key = }")
        st.sidebar.write(f"{session_state[key] = }")
    return result


# dict from param name to function showing this param
param2func = {
    "num_interval": select_num_interval,
    "several_nums": select_several_nums,
    "radio": select_radio,
    "rgb": select_RGB,
    "checkbox": select_checkbox,
    "min_max": select_min_max,
}

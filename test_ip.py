import streamlit as st
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    print("Has ctx:", ctx is not None)
    if hasattr(st, "context") and hasattr(st.context, "headers"):
        print("Headers:", dict(st.context.headers))
except Exception as e:
    print("Error:", e)

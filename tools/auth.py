import os
import tempfile
import streamlit as st


def setup_gcp_credentials():
    """
    Makes Vertex / Veo auth work on Streamlit Cloud.
    """

    # Already set (local dev / docker / CI)
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    if "SERVICE_ACCOUNT_JSON" not in st.secrets:
        raise RuntimeError("SERVICE_ACCOUNT_JSON missing from Streamlit secrets")

    # Write secret JSON to temp file
    tmp_dir = tempfile.gettempdir()
    cred_path = os.path.join(tmp_dir, "service_account.json")

    with open(cred_path, "w") as f:
        f.write(st.secrets["SERVICE_ACCOUNT_JSON"])

    # Export env vars for Google SDKs
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    os.environ["VERTEX_PROJECT"] = st.secrets["VERTEX_PROJECT"]
    os.environ["VERTEX_LOCATION"] = st.secrets["VERTEX_LOCATION"]

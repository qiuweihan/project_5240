import streamlit as st
from backend import run_pipeline


def main():
    st.set_page_config(page_title="Ticket Support System", layout="centered")

    st.title("Ticket Support System")
    st.write("Enter a customer support message below to classify the ticket and generate a support note.")

    user_input = st.text_area("Customer Message", height=180)

    if st.button("Run Pipeline"):
        if user_input.strip():
            result = run_pipeline(user_input)

            st.subheader("Predicted Category")
            st.write(result["top_category"])

            st.subheader("Top-3 Categories")
            for item in result["top_categories"]:
                st.write(f"{item['category']} — {item['probability']:.4f}")

            st.subheader("Support Note")
            st.write(result["support_note"])

            st.subheader("Suggested Reply Template")
            st.write(result["reply_template"])
        else:
            st.warning("Please enter a customer support message.")


if __name__ == "__main__":
    main()

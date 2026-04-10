import streamlit as st
import pandas as pd
from google import genai
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import base64
from io import BytesIO



st.set_page_config(
    page_title="Ask Your CSV",
    page_icon="📊",
    layout="wide"
)


# Initialize Gemini client

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=st.secrets["Gemini_API_Key"])

# Helper function for export
def export_conversation():
    """Export conversation history as HTML (works like PDF when printed)"""
    if not st.session_state.messages:
        return None
    
    # Create HTML content with embedded styles
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            h3 {{ color: #888; margin-top: 20px; }}
            .question {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .answer {{ padding: 10px; margin: 10px 0; }}
            .metadata {{ color: #999; font-size: 14px; }}
            code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis Report</h1>
        <p class="metadata">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    """
    
    # Add data summary
    if st.session_state.df is not None:
        html_content += f"""
        <h2>Dataset Information</h2>
        <ul>
            <li>Total Rows: {st.session_state.df.shape[0]}</li>
            <li>Total Columns: {st.session_state.df.shape[1]}</li>
            <li>Column Names: {', '.join(st.session_state.df.columns)}</li>
        </ul>
        """
    
    # Add conversation
    html_content += "<h2>Analysis Conversation</h2>"
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            html_content += f'<div class="question"><strong>Question:</strong> {msg["content"]}</div>'
        else:
            # Convert markdown code blocks to HTML
            content = msg["content"].replace("```python", "<pre><code>").replace("```", "</code></pre>")
            html_content += f'<div class="answer"><strong>Analysis:</strong><br>{content}</div>'
            if "figure" in msg:
                html_content += '<p><em>[Visualization generated - see application for details]</em></p>'
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content


# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None




st.title("📊 Ask Your CSV")
st.markdown("Upload your data and ask questions in plain English!")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_summary = {
                "shape": df.shape,
                "columns": df.columns.astype(str).tolist(),
                "dtypes": df.dtypes.astype(str).tolist(),
                "sample": df.head().to_string(),
                "stats": df.describe().to_string()
            }
            st.success(f"✅ :Loaded {df.shape[0]} rows x {df.shape[1]} columns")

            # Data preview
            with st.expander('Preview Data'):
                st.dataframe(df.head())

            # Basic Stats
            with st.expander('Basic Stats'):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                with col2:
                    st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.2f} KB")
                    st.metric("Missing Values", df.isnull().sum().sum())

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info(("Please make sure your file is a valid CSV format."))

        else:
            st.info("👆 Upload a CSV file to start analyzing!")

        # Export options (only show if there are messages)
        if st.session_state.messages:
            st.sidebar.markdown("---")
            st.sidebar.header("💾 Export Options")
            if st.sidebar.button("Generate Report"):
                export_html = export_conversation()
                st.sidebar.download_button(
                    label="📥 Download Report (HTML)",
                    data=export_html,
                    file_name=f"data_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html"
                )
                st.sidebar.info("💡 Tip: Open the HTML file and print to PDF for best results")


# Main chat interface
if st.session_state.df is not None:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-display any saved figures
            if 'figure' in msg:
                st.pyplot(msg['figure'])

    # Chat input
    user_input = st.chat_input('Ask a question about your data')

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        #Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare data context with token optimisation
        df = st.session_state.df
        if len(df) > 100:
            data_context = f"""
            Dataset shape: {st.session_state.data_summary['shape']}
            Columns: {';'.join(st.session_state.data_summary['columns'])}
            Data types: {st.session_state.data_summary['dtypes']}
            Sample rows: {st.session_state.data_summary['sample']}
            Basic Statistics: {st.session_state.data_summary['stats']}
            """
        else:
            data_context = f"""
            Dataset:
            {df.to_string()}
            """

        # system prompt
        system_prompt = f""" You are a helpful Data analyst.

        The user has uploaded a CSV file with the following information: {data_context}

        The data is loaded in a pandas dataframe called 'df'.

        Guidelines:
        - Answer the user's questions clearly and concisely
        - If the question requires analysis, write Python code using pandas, matplotlib, or seaborn
        - For visualizations always use plt.figure() before plotting and include plt.tight_layout()
        - Always validate data before operations (check for nulls, data types, etc.) 
        - If you can't answer due to data limitations, explain why
        - Keep responses focused on the data and the questions asked
        
        When you're writing code:
        - CRITICAL: DO NOT attempt to guess, estimate, or hallucinate numeric results or data findings in your text response before you run the code! Once you get the results then state your answer.
        - Import statements are already done (pandas as pd, matplotlib.pyplot as plt, seaborn as sns)
        - The dataframe is available as 'df'
        - For plots, use plt.figure(figsize=(10,6)) for better display
        - Always add titles and labels to plots
        """

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing..."):
                try:
                    # Format conversation history for Gemini
                    # Slicing with [-6:] limits the context to only the last 3 exchanges!
                    history = []
                    for msg in st.session_state.messages[-6:]:
                        # Truncate long message in history to save tokens
                        content = msg["content"]
                        if len(content) > 500:
                            content = content[:500] + "...[TRUNCATED]..."
                            
                        gemini_role = "model" if msg["role"] == "assistant" else "user"
                        history.append({"role": gemini_role, "parts": [{"text": content}]})

                    response = client.models.generate_content(
                        model="gemini-3.1-pro-preview",
                        contents=history,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.1,
                            max_output_tokens=1500,
                        )
                    )
                    reply = response.text
                    message_placeholder.markdown(reply)

                    # try to execute any code in the response
                    figure_generated = None
                    if "```python" in reply:
                        code_block = reply.split("```python")
                        # Execute the first python block generated
                        code = code_block[1].split("```")[0]
                            
                        try:
                            # Capture warnings
                            with warnings.catch_warnings(record=True) as w:
                                warnings.simplefilter("always")
                            # create figure for potential plots
                            plt.figure(figsize=(10,6))

                            # execute code in a controlled environment
                            exec_globals = {
                                'df': df,
                                'pd': pd,
                                'plt': plt,
                                'sns': sns,
                                'st': st
                            }

                            
                            exec(code.strip(), exec_globals)
                                
             
                            # Display warnings
                            if w:
                                for warning in w:
                                    st.info(f"Note: {warning.message}")

                            # Display the plot if created
                            fig = plt.gcf()
                            if fig.get_axes():
                                st.pyplot(fig)
                                figure_generated = fig
                            
                            # Clean up and close all figures to prevent memory leaks in streamlit
                            plt.close('all')

                        except Exception as e:
                            error_type = type(e).__name__
                            st.error(f"Error executing code: {error_type}")
                            
                            # Provide helpful context based on error type
                            if "NameError" in str(e):
                                st.info("This might mean a column name is misspelled or doesn't exist.")
                            elif "TypeError" in str(e):
                                st.info("This often happens when trying to plot non-numeric data.")
                            elif "KeyError" in str(e):
                                st.info("The specified column might not exist in your dataset.")
                            else:
                                st.info("Try rephrasing your question or check your data format.")
                            
                            st.code(code, language="python")

                    # Save assistant response to history ONCE at the end
                    msg_obj = {"role": "assistant", "content": reply}
                    if figure_generated:
                        msg_obj['figure'] = figure_generated
                    st.session_state.messages.append(msg_obj)

                except genai.errors.APIError as e:
                    st.error(f"Gemini API Error: {str(e)}")
                    st.info("Please check your API key and try again.")
                    

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Please try again or rephrase your question.")



else:
    # No data uploaded state
    col1, col2, col3 = st.columns(3)
    with col2:
        st.info("👆 Upload a CSV file to start analyzing!")

        # Example questions
        st.markdown("### 💡 Example questions you can ask:")
        st.markdown("""
        - What are the main trends in my data?
        - Show me a correlation matrix
        - Create a bar chart of the top 10 categories
        - What's the average value by month?
        - Are there any outliers iin the price column?
        """)
    
# footer with helpful tips
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey; font-size: 12px;">
💡 Tip: Be specific with your questions for better results! |
🔒 Your data stays private and is not stored
</div>
""", unsafe_allow_html=True)
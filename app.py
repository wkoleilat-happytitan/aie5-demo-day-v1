import os
import asyncio
import subprocess
from pathlib import Path
import chainlit as cl
from capability_extractor import create_retriever_from_file, extract_capability
from open_deep_research_buildorbuy_py import run_graph_and_show_report

# Chat initialization
@cl.on_chat_start
async def start():
    cl.user_session.set("file_received", False)
    cl.user_session.set("capability_extracted", False)
    
    welcome_message = """
    👋 Welcome to the Build or Buy Assistant!

    This tool helps you make informed decisions about whether to build a system or a capability in-house or buy an existing solution.

    How it works:
    1. Upload a document containing your user stories or technical specifications
    2. The assistant will analyze the document to understand the core capability
    3. You'll receive a detailed build or buy analysis including:
       - Market research on existing solutions
       - Cost comparisons
       - Implementation considerations
       - Recommendations based on your specific needs

    Let's get started! Please upload your document (PDF, TXT, or MD format) describing the user stories or technical specifications.
    """
    
    await cl.Message(content=welcome_message).send()
    
    try:
        files = await cl.AskFileMessage(
            content="📄 Upload your requirements document (PDF, TXT, or MD format)",
            accept=["text/plain", "application/pdf", ".md"],
            max_size_mb=20,
            timeout=180,
        ).send()

        if files:
            file = files[0]
            try:
                retriever = create_retriever_from_file(file.path)
                result = extract_capability(retriever)
                
                await cl.Message(
                    content=f"""
## Analysis Results

### System Description
{result['system_description']}

### Capability
**{result['capability']}**
                    """
                ).send()

                cl.user_session.set("capability", result['capability'])
                cl.user_session.set("file_received", True)
                await asyncio.sleep(1)
                
                await cl.Message(
                    content=f"Would you like to run a build or buy analysis report for this capability: **{result['capability']}**?",
                    actions=[
                        cl.Action(name="run_analysis", value="yes", label="✅ Run Analysis", payload={"action": "run"}),
                        cl.Action(name="cancel_analysis", value="no", label="❌ Cancel", payload={"action": "cancel"})
                    ]
                ).send()
            except Exception as e:
                await cl.Message(f"Error processing file: {str(e)}").send()
    except Exception as e:
        await cl.Message(f"Error receiving file: {str(e)}").send()

# Action handlers
@cl.action_callback("run_analysis")
async def on_run_analysis(action):
    capability = cl.user_session.get("capability")
    await cl.Message("🔍 Starting the build or buy analysis report...").send()
    
    try:
        result = await run_graph_and_show_report(capability=capability, auto_approve_plan=True)
        
        if isinstance(result, dict) and 'final_report' in result:
            report_content = result['final_report']
            await cl.Message(content=f"# Build vs Buy Analysis Report: {capability}\n\n{report_content}").send()
            
            try:
                pdf_file = await convert_to_slides_and_get_file(report_content, capability)
                file_element = cl.File(name=pdf_file.name, path=str(pdf_file), display="inline")
                await cl.Message(content="📊 Download the slide deck version of the report", elements=[file_element]).send()
                pdf_file.unlink(missing_ok=True)
            except Exception as e:
                await cl.Message(content=f"⚠️ Could not generate PDF slides: {str(e)}").send()
        else:
            await cl.Message(content="❌ Error: Could not generate final report.").send()
    except Exception as e:
        await cl.Message(f"❌ Error during analysis: {str(e)}").send()

@cl.action_callback("cancel_analysis")
async def on_cancel_analysis(action):
    await cl.Message("👋 Thank you for using the app!").send()

# Slide generation utilities
async def convert_to_slides_and_get_file(report_content: str, capability: str) -> Path:
    slides_dir = Path("slides")
    slides_dir.mkdir(exist_ok=True)
    
    filename = f"build_vs_buy_{capability.lower().replace(' ', '_')}"
    md_file = slides_dir / f"{filename}.md"
    pdf_file = slides_dir / f"{filename}.pdf"
    
    def extract_section(content: str, section_name: str) -> str:
        try:
            if section_name not in content:
                return "Section not found"
            sections = content.split(section_name)
            if len(sections) < 2:
                return "Section not found"
            section_content = sections[1].split("\n## ")[0].strip()
            return section_content
        except Exception as e:
            print(f"Error extracting {section_name}: {str(e)}")
            return f"Error extracting {section_name} section"
    
    def get_sections(content: str) -> list[str]:
        sections = []
        for line in content.split('\n'):
            if line.startswith('## '):
                sections.append(line.replace('## ', '').strip())
        return sections
    
    sections = get_sections(report_content)
    
    slides = [f"""---
marp: true
theme: default
paginate: true
---

# Build vs Buy Analysis: {capability}
<!-- _class: lead -->"""]
    
    for section in sections:
        slides.append(f"""
---

## {section}

{extract_section(report_content, section)}""")
    
    md_content = "\n".join(slides)
    md_file.write_text(md_content)
    
    try:
        subprocess.run(["marp", str(md_file), "--pdf", "-o", str(pdf_file)], check=True)
        return pdf_file
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        raise
    finally:
        md_file.unlink(missing_ok=True)
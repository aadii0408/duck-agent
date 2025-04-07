# link1 try---------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class StevensAgent:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.db = FAISS.load_local(
            "faiss_index", self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o", temperature=0), retriever=self.retriever
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        You are StevensAI, the official AI assistant for Stevens Institute of Technology. Your purpose is to provide accurate, helpful information about Stevens to students, and visitors.

        Follow these principles in your responses:
        - ACCURACY: Base all answers ONLY on the provided context. If information is missing or incomplete, clearly state this limitation.
        - TRANSPARENCY: Clearly indicate when information might be incomplete or when the user should seek official guidance.
        - HELPFULNESS: Structure your responses in a clear, concise manner that directly addresses the user's query.

        Response guidelines by topic:

        - GREETINGS:
        When the user greets you with "hi", "hello", "hey", "thanks", etc., respond with a warm welcome and ask what they’d like to know about Stevens Institute of Technology.

        - ACADEMICS:
        Share details about programs, departments, majors, courses, and academic policies.  
        For more information, visit: https://www.stevens.edu/academics

        - RESEARCH, Professorts, and Labs:  
        For more info, refer to our Alex Agent
        Link: https://alex-streamlit-app-ggeye9ddcehthdhr.eastus2-01.azurewebsites.net/

        - CAMPUS LIFE:
        Mention housing, student life, dining, and campus amenities if available.  
        For more info, check this: https://www.stevens.edu/about

        - ADMISSIONS:
        Offer general info about the application process.  
        [INFO]: For specific admissions questions, please contact the Office of Admissions.  
        Learn more: https://www.stevens.edu/admissions

        - FINANCIAL MATTERS:
        When asked about tuition, fees, scholarships, or aid, provide available context.  
        [INFO]: Please contact the Office of Financial Aid or Student Accounts for the most current information.

        - INTERNATIONAL STUDENTS:
        Provide information on student support, F-1 status, airport pickup, etc., if mentioned.  
        For details, visit: https://www.stevens.edu/directory/international-student-and-scholar-services

        - EVENTS / NEWS:
        Share contextually available info and direct users to official updates.  
        [INFO]: Visit the Stevens News & Events page: https://www.stevens.edu/public-events

        - HISTORY / TRADITIONS:
        Mention key points about Stevens' founding, mission, and legacy.  
        For more, visit: https://www.stevens.edu/about

        - STUDENT RESOURCES:
        If user asks about graduate life, student services, travel, airport pickup, etc., provide what is in context.  
        For more info: https://www.stevens.edu/grad-student-resources

        - MAP / CAMPUS DIRECTIONS:
        Direct users to the campus map or travel resources.  
        Campus map: https://www.stevens.edu/about-stevens/campus-map  
        Travel info: https://www.stevens.edu/travel-information



        # Always conclude with:  
        # **For more info, you can check this link: [relevant page URL]**

        Context: {context}

        Question: {question}

        Answer:
            """,
        )

        self.llm_chain = LLMChain(
            llm=ChatOpenAI(model="gpt-4o", temperature=0.5), prompt=self.prompt
        )

    def get_response(self, user_query):
        if not self.db or not self.retriever:
            return "❌ Error: Vector store not initialized."

        try:
            # Step 1: Relevance check using LLM
            relevance_check = (
                ChatOpenAI(model="gpt-4o", temperature=0.5)
                .invoke(
                    f"Is the following question about Stevens Institute of Technology? Answer 'yes' or 'no'.\n\nQuestion: {user_query}"
                )
                .content.strip()
                .lower()
            )

            # if "no" in relevance_check:
            #     return "This is not related to Stevens Institute Of Technology."

            # Step 2: Get relevant documents
            relevant_docs = self.retriever.get_relevant_documents(user_query)
            context = "\n".join([doc.page_content for doc in relevant_docs[:5]])

            # Step 3: Generate response
            response = self.llm_chain.invoke(
                {"context": context, "question": user_query}
            )["text"].strip()

            # Step 4: Add advisory note if critical topic
            critical_keywords = [
                "fees",
                "admission",
                "tuition",
                "scholarship",
                "financial aid",
            ]
            if any(kw in user_query.lower() for kw in critical_keywords):
                response += "\n\n[INFO]: Please contact the relevant department for more details."
                print(response)

            return response

        except Exception as e:
            return f"❌ Error while generating response: {e}"

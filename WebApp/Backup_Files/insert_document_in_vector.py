from langchain_core.documents import Document
from RAG import VectorStorage

# Initialize the vector store
vector_store = VectorStorage()

# Dummy policy documents (replace with real content in practice)
dummy_policies = [
    "Privacy Policy: We value your privacy and aim to protect your personal information. This policy outlines how we collect, store, and use your data. Your personal data will not be shared without consent.",
    "Code of Conduct: Employees must adhere to ethical standards and promote respect, integrity, and fairness in the workplace. Harassment, discrimination, and unethical behavior will not be tolerated.",
    "Employee Benefits: Our company offers a range of benefits, including health insurance, paid time off, retirement plans, and professional development opportunities. Full-time employees are eligible for all benefits.",
    "Anti-Discrimination Policy: We are committed to providing a workplace free from discrimination. We do not tolerate any form of discrimination based on race, gender, age, disability, or sexual orientation.",
    "Equal Opportunity Employment: We are an equal opportunity employer. We ensure that all hiring decisions are made based on merit and qualifications, without regard to race, color, religion, sex, or national origin.",
    "Workplace Safety Policy: The company is committed to providing a safe work environment. Employees must follow safety protocols and report any unsafe conditions immediately to the safety officer.",
    "Remote Work Policy: Employees are permitted to work remotely in accordance with the company's remote work guidelines. All remote work arrangements must be approved by a manager.",
    "Leave of Absence Policy: Employees are entitled to leave in case of illness, family emergencies, or personal matters. Documentation may be required depending on the length and nature of the leave.",
    "Confidentiality Agreement: Employees must maintain the confidentiality of sensitive company information. Breaching confidentiality could result in disciplinary action, including termination of employment.",
    "Harassment Prevention Policy: We do not tolerate harassment in the workplace. Any complaints of harassment will be thoroughly investigated, and corrective action will be taken if necessary.",
    "Compensation Policy: Employees will be compensated based on their role, experience, and performance. Compensation reviews occur annually, with salary increases based on individual performance and company growth.",
    "Performance Review Policy: Employees will undergo annual performance reviews, during which their contributions, goals, and areas for improvement will be discussed with their managers.",
    "Termination Policy: Employment may be terminated by either the employer or the employee, with appropriate notice provided. In cases of misconduct, immediate termination may occur.",
    "Social Media Policy: Employees must ensure their social media presence does not conflict with the company's values or reputation. Confidential information must not be shared on social media.",
    "Dress Code Policy: Employees are expected to adhere to a professional dress code while in the office. Clothing should be business casual unless otherwise specified for certain roles.",
    "Expense Reimbursement Policy: Employees are entitled to reimbursement for certain business-related expenses, such as travel, meals, and supplies. All expenses must be pre-approved and substantiated with receipts.",
    "Intellectual Property Policy: Any work created by employees during their employment, including inventions, designs, and content, will be considered the property of the company.",
    "Health and Wellness Program: The company offers a range of health and wellness initiatives, including gym memberships, mental health resources, and workshops aimed at promoting employee well-being.",
    "Environmental Responsibility Policy: We are committed to reducing our environmental impact. Employees are encouraged to reduce waste, recycle, and participate in sustainability initiatives.",
    "Customer Service Policy: We strive to provide excellent customer service by addressing customer needs promptly and professionally. Feedback from customers is valued and used to improve our services."
]

# Convert each policy to a LangChain Document object
documents = [Document(page_content=policy) for policy in dummy_policies]

# Store the documents into the vector store
vector_store.store_embeddings(
    thread_id="60306e06-822a-444a-a0c3-dc8bc488231f",
    documents=documents
)

print("Documents have been successfully stored!")

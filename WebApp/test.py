from utils import extract_data_from_resume , parse_file




if __name__ == "__main__":

    resume_data = parse_file(file_path="Uploads/data_science_resume.pdf")


    print(resume_data)


    extracted_data = extract_data_from_resume(
        resume_data=resume_data
    )

    print(extracted_data)


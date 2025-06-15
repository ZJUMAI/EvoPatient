# EvoPatient
The code repository for our paper [LLMs Can Simulate Standardized Patients via Agent Coevolution](https://arxiv.org/abs/2412.11716). We are currently in the process of preparing the data and code for release.

## 🎥 Demo Video

<div align="center">
  <video width="800" controls>
    <source src="assets/demo_video.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

*EvoPatient system demonstration - showing standardized patient simulation through agent coevolution*

## 📖 Abstract
Training medical personnel using standardized patients (SPs) remains a complex challenge, requiring extensive domain expertise and role-specific practice. Previous research on Large Language Model (LLM)-based SPs mostly focuses on improving data retrieval accuracy or adjusting prompts through human feedback. However, this focus has overlooked the critical need for patient agents to learn a standardized presentation pattern that transforms data into human-like patient responses through unsupervised simulations. To address this gap, we propose EvoPatient, a novel simulated patient framework in which a patient agent and doctor agents simulate the diagnostic process through multi-turn dialogues, simultaneously gathering experience to improve the quality of both questions and answers, ultimately enabling human doctor training. Extensive experiments on various cases demonstrate that, by providing only overall SP requirements, our framework improves over existing reasoning methods by more than 10% in requirement alignment and better human preference, while achieving an optimal balance of resource consumption after evolving over 200 cases for 10 hours, with excellent generalizability.

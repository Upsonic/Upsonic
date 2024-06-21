# Introduction
In this example we have an idea to summerize whole day of an employee via GPT Computer Assistant. 



# Code
```console
computerassistant --api
```


```python
from gpt_computer_assistant.remote import remote



remote.profile("Screen Analysis")

# We will loop for 5 minutes

loop_results = []


for i in range(1000):
    remote.reset_memory()

    remote.just_screenshot()

    detailed_analyses = remote.input("What is in the scren, detailed analyses")
    app_name = remote.input("What is the app that the employee is using?")
    subject = remote.input("What is the subject of this usage of the app?")
    activity = remote.input("What is the employee doing now?")
    loop_results.append({"detailed_analyses": detailed_analyses, "app_name": app_name, "subject": subject, "activity": activity})
    

    remote.wait(10) 


# Summery of the work day

summery_results = []

remote.profile("Summerizer")
remote.reset_memory()
for i in loop_results:
    
    total_string = i["detailed_analyses"] + " " + i["app_name"] + " " + i["subject"] + " " + i["activity"]
    total_string = "Please summerize the work day" + total_string
    summerized = remote.input(total_string)
    summery_results.append(summerized)


print("Summery: ", summery_results)
    
```
# Automated Scheduling for Temporal Analysis

The Monitoring and Repeatability Framework is designed to support long-term assessment of deer impact by periodically re-analyzing imagery. The core logic is encapsulated in the `POST /api/v1/monitoring/temporal_analysis` endpoint.

Automating this process is an infrastructure-level task that depends on your specific deployment environment. Below are two common approaches to achieve this.

## Option 1: Using a Cron Job

A simple and effective method for scheduling tasks on Linux-based systems is using `cron`. You can set up a cron job to send a `POST` request to the API endpoint at regular intervals.

### Example

This example runs the temporal analysis at 2:00 AM every Monday, comparing the last 7 days to the 7 days prior to that.

1.  Open your crontab for editing:
    ```bash
    crontab -e
    ```

2.  Add the following line. You will need to adjust the dates and the endpoint URL for your specific needs. This example uses `curl` and `date` commands to dynamically set the date ranges.

    ```cron
    # Run temporal analysis every Monday at 2 AM
    0 2 * * 1 /usr/bin/curl -X POST "http://localhost:8000/api/v1/monitoring/temporal_analysis?start_date1=$(date -d '14 days ago' +\%Y-\%m-\%d)&end_date1=$(date -d '7 days ago' +\%Y-\%m-\%d)&start_date2=$(date -d '7 days ago' +\%Y-\%m-\%d)&end_date2=$(date +\%Y-\%m-\%d)"
    ```

### Pros and Cons
- **Pros:** Simple to set up, available on most Linux systems by default.
- **Cons:** Lacks advanced features like retry mechanisms, logging, and a UI for monitoring. Can be brittle if the command becomes complex.

## Option 2: Using a Task Queue (e.g., Celery)

For production environments and more complex scheduling requirements, a dedicated task queue with a scheduler component is the recommended approach. [Celery](https://docs.celeryq.dev/en/stable/index.html) is a popular choice in the Python ecosystem.

This would involve:
1.  **Setting up Celery:** Add Celery to the project, configure a message broker (like RabbitMQ or Redis), and define a Celery app.
2.  **Creating a Task:** Wrap the `services.temporal_analysis` function call in a Celery task.
3.  **Scheduling the Task:** Use [Celery Beat](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html), Celery's built-in periodic task scheduler, to run the task at your desired interval.

### Example Snippet (Conceptual)

```python
# In a new file, e.g., app/tasks.py
from celery import Celery
from celery.schedules import crontab
from app.monitoring.services import temporal_analysis
import datetime

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Run every Monday at 2 AM
    sender.add_periodic_task(
        crontab(hour=2, minute=0, day_of_week=1),
        run_weekly_analysis.s(),
    )

@app.task
def run_weekly_analysis():
    end_date = datetime.date.today()
    start_date2 = end_date - datetime.timedelta(days=7)
    end_date1 = start_date2
    start_date1 = end_date1 - datetime.timedelta(days=7)

    temporal_analysis(
        start_date1=start_date1.strftime("%Y-%m-%d"),
        end_date1=end_date1.strftime("%Y-%m-%d"),
        start_date2=start_date2.strftime("%Y-%m-%d"),
        end_date2=end_date.strftime("%Y-%m-%d"),
    )
```

### Pros and Cons
- **Pros:** Highly robust and scalable. Provides logging, retries, monitoring, and complex scheduling options.
- **Cons:** Adds more dependencies and complexity to the project's infrastructure.

## Conclusion

The choice of scheduling system depends on your project's scale and reliability requirements. For simple, periodic analysis, a cron job is sufficient. For mission-critical, production-grade applications, Celery is the superior choice.

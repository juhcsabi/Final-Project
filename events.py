import collections
import datetime
import json
import os
import pprint

import requests

event_list = collections.defaultdict(dict)
death_list = collections.defaultdict(dict)
data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"

for month in range(1, 13):
    r = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for day in range(1, r[month - 1] + 1):
        month_str = ("0" if month < 10 else "") + str(month)
        day_str = ("0" if day < 10 else "") + str(day)
        if day not in event_list[month]:
            event_list[month][day] = []
            death_list[month][day] = []
        date = f"{month_str}/{day_str}"

        url = 'https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/all/' + date

        headers = {
            'Authorization': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIwNGYxYzhjYTFiYWUwYTg2MTMyNzZjOTY3N2ViMDVhOSIsImp0aSI6IjVkY2JjMDdhZDQzZDE4ODQ1MWZlZjE0ZmQ1OWU3ZDQ5ZjgxOTYwMWYxYmZiZmZlNzFhMDEwNDAwMjU3MGFkMTU1NTI4ZGUyNjY2ZjJlMmRhIiwiaWF0IjoxNzA3MTIyMjE2Ljk4NjA1MSwibmJmIjoxNzA3MTIyMjE2Ljk4NjA1NCwiZXhwIjozMzI2NDAzMTAxNi45ODQ1Miwic3ViIjoiNzQ5MDE5NDAiLCJpc3MiOiJodHRwczovL21ldGEud2lraW1lZGlhLm9yZyIsInJhdGVsaW1pdCI6eyJyZXF1ZXN0c19wZXJfdW5pdCI6NTAwMCwidW5pdCI6IkhPVVIifSwic2NvcGVzIjpbImJhc2ljIl19.nukXsg-2jLJIkMKNxTj3tw03AOJ__tCFgOkO-HYgoAW3rIJOXtL49Ff9xFFR59Bd2xf-ZN8X3RDWD1gNQ4zrHOiNHBga9nayDEjSpTAXoVWuXj5SyXAYKXuYd79rur-RryR1XB-x9miVipc72pPYOwbWibdwNq8A-HIqLRaHhsLXQkWEXZcQy1XHyPk4cZpbKj3QKDxJLI8KOzs7cTvbbuyWdkLLBrJpwAt2Lk8NzG15ADW2bdUsUyQy34W02nqItRLFpN1aSjlGnn0P_1TEZqS8NgizuAqxRU2d6_VuuDEy4UBaJSBnS4vW-Yce8TfTvZt2sUBYnA5d76CU9dOinkOD6qSqtlbhiZPllMRYPQSnxbLtjsGFPor1hbV8tlwGTBrcnXoqyiFTaUTXZyiJNaZ2MwsPofSl4VlCpWBX2bWtng_vjq8Yba1i_E9_usPYzemsl8F77D7U1HL3r-2u5Sa90Vtag-GrC7-_MHSrv9cUwCk7U35mzIs_wy2V-79ObJ-4bTzVLNdHeP4sotIIL_fXsls3xfGpHx7ql7i0S86D71O8LfcsyKBBDMeSdnkZciGsJaJozWtaHdJkXTj2ZLWYPsb4PIb0SyFW_dIfwFWFRq1rAHiVCE5JCFp2-zUYykfH5a1WXN1LSK17_QiwyXf9WcH3x61ckSZVZjxN718',
            'User-Agent': 'Thesis_JCs'
        }

        response = requests.get(url, headers=headers)
        data = response.json()
        day_events = []
        for event_type in ["selected", "events"]:
            #pprint.pprint(data["selected"][0])
            #pprint.pprint(data)
            while event_type not in data:
                print(date, event_type)
                response = requests.get(url, headers=headers)
                data = response.json()
            events = data[event_type]
            for event in events:
                if 2012 <= event['year'] <= 2022:
                    found = False
                    for (text, year) in event_list[month][day]:
                        if event["text"] == text and event["year"] == year:
                            found = True
                            break
                    if not found:
                        event_list[month][day].append([event['text'], event['year']])
                    print(event['text'], event['year'])
        for event_type in ["deaths"]:
            while event_type not in data:
                print(date, event_type)
                response = requests.get(url, headers=headers)
                data = response.json()
            events = data[event_type]
            for event in events:
                if 2012 <= event['year'] <= 2022:
                    found = False
                    for (text, year) in death_list[month][day]:
                        if event['text'].split(",")[0] == text and event["year"] == year:
                            found = True
                            break
                    if not found:
                        death_list[month][day].append([event['text'].split(",")[0], event['year']])

with open(os.path.join(data_path, "events2.json"), "wt") as f:
    json.dump(event_list, f)
with open(os.path.join(data_path, "deaths2.json"), "wt") as f:
    json.dump(death_list, f)

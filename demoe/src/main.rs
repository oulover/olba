
mod log_config;
mod route;
mod user_dao;

use chrono::{DateTime, Utc};
async fn root() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() {
    // Parses an RFC 3339 date-and-time string into a `DateTime<FixedOffset>` value.
    //
    // Parses all valid RFC 3339 values (as well as the subset of valid ISO 8601 values that are
    // also valid RFC 3339 date-and-time values) and returns a new [`DateTime`] with a
    // [`FixedOffset`] corresponding to the parsed timezone. While RFC 3339 values come in a wide
    // variety of shapes and sizes, `1996-12-19T16:39:57-08:00` is an example of the most commonly
    // encountered variety of RFC 3339 formats.
    //
    // Why isn't this named `parse_from_iso8601`? That's because ISO 8601 allows representing
    // values in a wide range of formats, only some of which represent actual date-and-time
    // instances (rather than periods, ranges, dates, or times). Some valid ISO 8601 values are
    // also simultaneously valid RFC 3339 values, but not all RFC 3339 values are valid ISO 8601
    // values (or the other way around).

    let start = "2024-07-14T03:10:00+00:00";
    let start_time: DateTime<Utc> = DateTime::parse_from_rfc3339(start)
        .expect("Failed to parse date")
        .with_timezone(&Utc);


    let end = "2024-07-15T03:10:00+00:00";

    let end_time: DateTime<Utc> = DateTime::parse_from_rfc3339(end)
        .expect("Failed to parse date")
        .with_timezone(&Utc);




    let r =  calculate_task_status_now(LeadTaskStatus::Init,start_time,end_time);

    //println!("{:?}",r);
}

pub fn calculate_task_status_now(task_status: LeadTaskStatus, start_date: DateTime<Utc>, dead_line: DateTime<Utc>) -> LeadTaskStatus {
    let now = Utc::now();
    match task_status {
        LeadTaskStatus::Done => { LeadTaskStatus::Done }
        LeadTaskStatus::Close => { LeadTaskStatus::Close }
        _ => {
            if now.lt(&start_date){
                LeadTaskStatus::InFuture
            }else {
                if now.gt(&dead_line){
                    LeadTaskStatus::TimeOut
                }else {
                    LeadTaskStatus::Todo
                }
            }
        }
    }
}


#[derive(Debug)]
pub enum LeadTaskStatus{
    Init=0,
    InFuture=1,
    Todo=2,
    Done=3,
    TimeOut=4,
    Close=5,
}



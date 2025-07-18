class Respiration < ApplicationRecord
  belongs_to :user

  store_accessor :data, :summaryId, :startTimeInSeconds
  filterrific(
    default_filter_params: { 
      with_calendar_date_gte: Date.today,
      with_calendar_date_lte: Date.today
    },
   available_filters: [
     :with_calendar_date_gte,
     :with_calendar_date_lte
   ]
  )

  scope :with_calendar_date_gte, ->(reference_time) {
    where("calendar_date >= ?", reference_time)
  }

  scope :with_calendar_date_lte, ->(reference_time) {
    where("calendar_date <= ?", reference_time)
  }
end

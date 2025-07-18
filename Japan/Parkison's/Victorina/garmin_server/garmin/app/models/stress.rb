class Stress < ApplicationRecord
  belongs_to :user

  store_accessor :data, :summaryId, :calendarDate, :timeOffsetStressLevelValues

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
	  where("(data->>'calendarDate')::timestamp >= ?", reference_time)
	}

	scope :with_calendar_date_lte, ->(reference_time) {
	  where("(data->>'calendarDate')::timestamp <= ?", reference_time)
	}
end

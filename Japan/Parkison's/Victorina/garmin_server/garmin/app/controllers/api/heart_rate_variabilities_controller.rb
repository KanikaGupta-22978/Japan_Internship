module Api
	class HeartRateVariabilitiesController < ApiController
    def index
      heart_rate_variability_params.each do |data|
        # Check if incoming data can be associated with userId
        user = User.find_by(uid: data[:userId])
        if user.nil?
          # Check if incoming data can be associated with UAT
          # Default, since userId is not provided during OAuth)
          user = User.find_by(user_access_token: data[:userAccessToken])
          if user.nil?
            user = User.create(
              provider: 'garmin',
              uid: data[:userId],
              user_access_token: data[:userAccessToken]
            )
          else
            user.update(uid: data[:userId])
          end
        end

        calendar_date = Time.zone.at(data["startTimeInSeconds"]).to_date
        heart_rate_variabilities = user.heart_rate_variabilities.where(
          calendar_date: calendar_date
        )
        if heart_rate_variabilities.empty?
          user.heart_rate_variabilities.create(data: [data], calendar_date: calendar_date)
        elsif heart_rate_variabilities.count == 1
          heart_rate_variabilities.first.data = ( [data] + heart_rate_variabilities.first.data).uniq { |h| h["summaryId"] }
          heart_rate_variabilities.first.save
        else
          # TODO Add merging code when there are multiple records of same calendarDate
        end
      end

      json_response message: 'ok'
    end

    private
      def heart_rate_variability_params
        params.require(:hrv)
      end
  end
end

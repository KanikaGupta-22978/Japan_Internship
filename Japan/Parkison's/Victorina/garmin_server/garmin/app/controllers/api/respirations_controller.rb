module Api
  class RespirationsController < ApiController
    def index
      respirations_params.each do |data|
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
        respirations = user.respirations.where(
          calendar_date: calendar_date
        )
        if respirations.empty?
          user.respirations.create(data: [data], calendar_date: calendar_date)
        elsif respirations.count == 1
          respirations.first.data = ( [data] + respirations.first.data).uniq { |h| h["summaryId"] }
          respirations.first.save
        else
          # TODO Add merging code when there are multiple records of same calendarDate
        end
      end
      
      json_response message: 'ok'
    end

    private
      def respirations_params
        params.require(:allDayRespiration)
      end
  end
end
module Api
  class PulseOxesController < ApiController
    def index
      pulse_ox_params.each do |data|
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

        pulse_oxes = user.pulse_oxes.where("data->>'calendarDate' = ?", data[:calendarDate])
        if pulse_oxes.empty?
          user.pulse_oxes.create(data: data)
        elsif pulse_oxes.count == 1
          pulse_oxes.first.data.merge!(data.permit!)
          pulse_oxes.first.save
        # else
        #   TODO Add merging code when there are multiple records of same calendarDate
        end
      end

      json_response message: 'ok'
    end

    private
      def pulse_ox_params
        params.require(:pulseox)
      end
  end
end
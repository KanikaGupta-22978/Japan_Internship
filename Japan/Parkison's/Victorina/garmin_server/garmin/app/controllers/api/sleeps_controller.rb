module Api
	class SleepsController < ApiController
		def index
			sleep_params.each do |data|
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

				sleeps = user.sleeps.where("data->>'calendarDate' = ?", data[:calendarDate])
				if sleeps.empty?
					user.sleeps.create(data: data)
				elsif sleeps.count == 1
					sleeps.first.data.merge!(data.permit!)
					sleeps.first.save
				else
					# TODO Add merging code when there are multiple records of same calendarDate
				end
			end

			json_response message: 'ok'
		end

		private
			def sleep_params
				params.require(:sleeps)
			end
	end
end

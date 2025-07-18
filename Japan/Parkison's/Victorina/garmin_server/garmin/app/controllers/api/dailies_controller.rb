module Api
	class DailiesController < ApiController
		def index
			daily_params.each do |data|
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

				dailies = user.dailies.where("data->>'calendarDate' = ?", data[:calendarDate])
				if dailies.empty?
					user.dailies.create(data: data)
				elsif dailies.count == 1
					dailies.first.data.merge!(data.permit!)
					dailies.first.save
				else
					# TODO Add merging code when there are multiple records of same calendarDate
				end
			end

			json_response message: 'ok'
		end

		private
			def daily_params
				params.require(:dailies)
			end
	end
end

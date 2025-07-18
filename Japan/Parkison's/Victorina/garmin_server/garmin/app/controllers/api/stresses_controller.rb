module Api
	class StressesController < ApiController
		def index
			stress_params.each do |data|
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

				stresses = user.stresses.where("data->>'calendarDate' = ?", data[:calendarDate])
				if stresses.empty?
					user.stresses.create(data: data)
				elsif stresses.count == 1
					stresses.first.data.merge!(data.permit!)
					stresses.first.save
				else
					# TODO Add merging code when there are multiple records of same calendarDate
				end
			end

			json_response message: 'ok'
		end

		private
			def stress_params
				params.require(:stressDetails)
			end
	end
end

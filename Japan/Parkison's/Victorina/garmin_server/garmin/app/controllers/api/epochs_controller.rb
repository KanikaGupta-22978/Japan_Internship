module Api
	class EpochsController < ApiController
		def index
			epoch_params.each do |data|
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
				epochs = user.epochs.where(
					calendar_date: calendar_date
				)
				if epochs.empty?
					user.epochs.create(data: [data], calendar_date: calendar_date)
				elsif epochs.count == 1
					epochs.first.data = ( [data] + epochs.first.data).uniq { |h| h["summaryId"] }
					epochs.first.save
				else
					# TODO Add merging code when there are multiple records of same calendarDate
				end
			end

			json_response message: 'ok'
		end

		private
			def epoch_params
				params.require(:epochs)
			end
	end
end

class UsersController < ApplicationController
	before_action :check_environment
	before_action :authenticate_user!
  before_action :set_user, only: [:show]
  
  def index
  end

  def show
    ## HEART RATE
    @heart_rate_filterrific = initialize_filterrific(
      Daily,
      params[:filterrific]
    ) or return
    @dailies = @heart_rate_filterrific.find.where(user: @user)

    @heart_rate_chart_data = @dailies.map { |daily|
      daily.timeOffsetHeartRateSamples.transform_keys do |seconds|
        Time.zone.parse(daily.calendarDate) + seconds.to_i.seconds
      end
    }.flatten.reduce Hash.new, :merge

    ## SLEEP
    @sleep_filterrific = initialize_filterrific(
      Sleep,
      params[:filterrific]
    ) or return
    @sleeps = @sleep_filterrific.find.where(user: @user)

    @sleep_chart_data = @sleeps.map { |sleep|
      {
        calendar_date: sleep.calendarDate,
        sleep_chart_data: sleep.sleepLevelsMap.map { |sleep_type, durations| 
          durations.map { |duration| 
            [ sleep_type, Time.zone.at(duration["startTimeInSeconds"]), Time.zone.at(duration["endTimeInSeconds"]) ]
          }
        }.flatten(1).sort_by { |d| d[1] }
      }
    }.sort_by { |d| d[:calendar_date] }

    ## STRESS
    @stress_filterrific = initialize_filterrific(
      Stress,
      params[:filterrific]
    ) or return
    @stresses = @stress_filterrific.find.where(user: @user)

    @stress_chart_data = @stresses.map { |stress|
      stress.timeOffsetStressLevelValues.transform_keys do |seconds|
        Time.zone.parse(stress.calendarDate) + seconds.to_i.seconds
      end
    }.flatten.reduce Hash.new, :merge

    @grouped_stress_chart_data = @stress_chart_data.group_by { |stress_chart_data|
      helpers.interpret_stress( stress_chart_data[1] )
    }.map { |k,v| 
      {
        name: k,
        data: v
      }
    }

    ## STEPS
    @epoch_filterrific = initialize_filterrific(
      Epoch,
      params[:filterrific]
    ) or return
    @epochs = @epoch_filterrific.find.where(user: @user)

    @epoch_chart_data = @epochs.map { |epoch|
      epoch.data.map { |d|
        { Time.zone.at(d["startTimeInSeconds"]) => d["steps"] }
      }
    }.flatten.sort_by { |d| d[0] }.reduce(Hash.new, :merge)

    ## BLOOD OXYGEN PULSE OX
    @pulse_oxes_filterrific = initialize_filterrific(
      PulseOx,
      params[:filterrific]
    ) or return
    @pulse_oxes = @pulse_oxes_filterrific.find.where(user: @user)

    @pulse_ox_chart_data = @pulse_oxes.map { |pulse_ox|
      pulse_ox.timeOffsetSpo2Values.transform_keys do |seconds|
        Time.zone.parse(pulse_ox.calendarDate) + seconds.to_i.seconds
      end
    }.flatten.reduce Hash.new, :merge


    ## RESPIRATIONS
    @respirations_filterrific = initialize_filterrific(
      Respiration,
      params[:filterrific]
    ) or return
    @respirations = @respirations_filterrific.find.where(user: @user)

    @respiration_chart_data = @respirations.map { |respiration|
      respiration.data.map { |d|
        d["timeOffsetEpochToBreaths"].transform_keys do |seconds|
          Time.zone.at(d["startTimeInSeconds"]) + seconds.to_i.seconds
        end
      }
    }.flatten.sort_by { |d| d[0] }.reduce(Hash.new, :merge)

    # Order chart_datas by date in ascending order
    @heart_rate_chart_data = @heart_rate_chart_data.sort_by { |k,v| k }.to_h
    @stress_chart_data = @stress_chart_data.sort_by { |k,v| k }.to_h
    @epoch_chart_data = @epoch_chart_data.sort_by { |k,v| k }.to_h

    # Get the latest key from heart_rate_chart_data
    latest_dates = [
      @heart_rate_chart_data.keys.last,
      @stress_chart_data.keys.last,
      @epoch_chart_data.keys.last
    ]
    # # Check if latest_dates has nil values
    # if !latest_dates.include?(nil)
    #   # Get the minimum date from the latest dates, and round down to the nearest hour
    #   @timestamp = latest_dates.min.change(min: 0, sec: 0)
    #   @timestamp_ago = @timestamp - 1.day

    #   # Remove chart_data that is not within the range of the minimum date and the date given
    #   @heart_rate_chart_data = @heart_rate_chart_data.select { 
    #     |k,v| k >= @timestamp_ago && k <= @timestamp }
    #   @stress_chart_data = @stress_chart_data.select {
    #     |k,v| k >= @timestamp_ago && k <= @timestamp }
    #   @epoch_chart_data = @epoch_chart_data.select {
    #     |k,v| k >= @timestamp_ago && k <= @timestamp }
    # end

    # content = render_to_string template: 'users/show.xlsx.axlsx'
    # File.open("#{Rails.root}/fastapi/app/tmp.xlsx", "w+b") { |f| f.puts content }

    # # Get the host from ENV
    # url = "http://#{ENV['FASTAPI_URL']}:3002/process"
    # begin
		# 	result = Curl.get(url) do |http|
		# 	  http.timeout = 20
		# 	  http.ssl_verify_peer = false
		# 	end

		# 	if result.response_code == 200
		# 	  @forecasts = JSON.parse(result.body, symbolize_names: false)
		# 	  @forecasts = @forecasts["forecasts"].map { |k, v| 
		# 	  	# { Time.zone.parse(k).strftime("%b %-d, %l:%M%p") => v }
		# 	  	{ Time.zone.parse(k).strftime("%l:%M%p") => v }
		# 	  }.flatten.reduce Hash.new, :merge
		# 	else
		# 	  @forecasts = {
		# 	    error: "ERROR"
		# 	  }
		# 	end
    # rescue
    #   @forecasts = {
    #     error: "ERROR"
    #   }
    # end

  	respond_to do |format|
  		format.html
  		format.js
  		format.xlsx
  	end
  end

  def create
    @start_time = user_params[:start_time]
    @debugging_start_time = DateTime.parse(@start_time).in_time_zone(Time.zone).at_beginning_of_day
    @debugging_end_time = DateTime.parse(@start_time).in_time_zone(Time.zone).at_end_of_day

    @result = send_to_garmin(
      @debugging_start_time.to_i,
      @debugging_end_time.to_i
    )

    if @result.is_a?(Hash)
    else
      # To ensure that the shown data is only for the requested date
      @result = @result.select { |r|
        user_params[:start_time] == r[:calendarDate]
      }

      # For producing the chart data
      final = []
      @result.each do |summary|
        summary = summary.with_indifferent_access
        if @start_time == summary[:calendarDate]
          local_start_time = Time.at(summary[:startTimeInSeconds]).in_time_zone(Time.zone)
          final << summary[:timeOffsetHeartRateSamples].map do |seconds, heart_rate|
            { local_start_time + seconds.to_i => heart_rate }
          end
        end
      end
      @chart = final.flatten.reduce Hash.new, :merge
    end

    case params[:commit]
    when 'Download Excel' then
      render :users, formats: [:xlsx]
    else
      render :index
    end
  end

  private
    def check_environment
      if Rails.env.development? && current_user.nil?
        current_user = User.find(1)
        sign_in current_user
      end
    end
    
  	def user_params
  		params.require(:user).permit!
  	end

    def set_user
      @user = User.find_by(id: (params[:id] || params[:user_id]))
      if @user == current_user || current_user.id == 1
        @user = current_user if @user.nil?
      else
        redirect_to user_path(current_user), alert: "You're not allowed to visit this page."
      end
    end

    def send_to_garmin(start_time, end_time)
      consumer = OAuth::Consumer.new(
        ENV['GARMIN_KEY'],
        ENV['GARMIN_SECRET'],
        { site: ENV['GARMIN_URL'], http_method: :get }
      )

      access_token = OAuth::AccessToken.new(
        consumer,
        current_user.user_access_token,
        current_user.user_access_token_secret
      )

      parameter = {
        uploadStartTimeInSeconds: start_time,
        uploadEndTimeInSeconds: end_time
      }.to_query

      path = '/wellness-api/rest/dailies'

      begin
        result = access_token.get("#{path}?#{parameter}")

        if result.code == "200"
          data = JSON.parse(result.body, symbolize_names: true)
          if data.empty?
            return {
              error: I18n.t('errors.missing_data')
            }
          else
            return data
          end
        else
          return {
            error: JSON.parse(result.body, symbolize_names: true)
          }
        end
      rescue Curl::Err::TimeoutError => e
        return {
          error: I18n.t('errors.curl_timeout')
        }
      rescue JSON::ParserError => e
        return {
          error: I18n.t('errors.json_parser')
        }
      rescue Exception => e
        return {
          error: I18n.t('errors.exception')
        }
      end
    end
end
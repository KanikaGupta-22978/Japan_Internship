module Api
  class PredictsController < ApiController
    def index
      # This is an API endpoint to predict the user's wearing-off
      # This endpoint accepts ID and date as parameters
      # Then, gets the user's data from the database
      # If there is no data from the date given, get the last available 1-day's data

      # Check if 'id' & 'timestamp' are given
      if params[:id].nil? || params[:timestamp].nil?
        json_response message: 'Please provide user ID and timestamp'
        return
      end

      # Check if user exists
      @user = User.find_by(id: params[:id])
      if @user.nil?
        json_response message: 'User does not exist'
        return
      else

      end

      # Check if timestamp is valid date time
      begin
        @original_timestamp = DateTime.parse(params[:timestamp])
        # Round down timestamp to nearest hour
        @original_timestamp = @original_timestamp.change(min: 0, sec: 0)
        # Get 1 day before the date given
        @original_timestamp_ago = @original_timestamp - 1.day
      rescue
        json_response({
          message: 'Please provide a valid timestamp in the format of YYYY-MM-DD HH:MM:SS'
        })
        return
      end

      # Get the user's data from the database
      # Get Daily data from one day from date until the date given
      dailies = @user.dailies.with_calendar_date_gte(
        @original_timestamp_ago
      ).with_calendar_date_lte(
        @original_timestamp
      )
      if dailies.empty?
        @timestamp = @user.dailies.get_latest_calendar_date.first.data['calendarDate'].to_datetime
        @timestamp_ago = @timestamp - 1.day
        dailies = @user.dailies.with_calendar_date_gte(@timestamp_ago).with_calendar_date_lte(@timestamp)
      end
      @heart_rate_chart_data = dailies.map { |daily|
        daily.timeOffsetHeartRateSamples.transform_keys do |seconds|
          Time.zone.parse(daily.calendarDate) + seconds.to_i.seconds
        end
      }.flatten.reduce Hash.new, :merge

      # Get Sleep data from one day from date until the date given
      sleeps = @user.sleeps.with_calendar_date_gte(@timestamp_ago).with_calendar_date_lte(@timestamp)
      @sleep_chart_data = sleeps.map { |sleep|
        {
          calendar_date: sleep.calendarDate,
          sleep_chart_data: sleep.sleepLevelsMap.map { |sleep_type, durations| 
            durations.map { |duration| 
              [ sleep_type, Time.zone.at(duration["startTimeInSeconds"]), Time.zone.at(duration["endTimeInSeconds"]) ]
            }
          }.flatten(1).sort_by { |d| d[1] }
        }
      }.sort_by { |d| d[:calendar_date] }

      # Get Stress data from one day from date until the date given
      stresses = @user.stresses.with_calendar_date_gte(@timestamp_ago).with_calendar_date_lte(@timestamp)
      @stress_chart_data = stresses.map { |stress|
        stress.timeOffsetStressLevelValues.transform_keys do |seconds|
          Time.zone.parse(stress.calendarDate) + seconds.to_i.seconds
        end
      }.flatten.reduce Hash.new, :merge

      
      # Get Steps data from one day from date until the date given
      epochs = @user.epochs.with_calendar_date_gte(@timestamp_ago).with_calendar_date_lte(@timestamp)
      @epoch_chart_data = epochs.map { |epoch|
        epoch.data.map { |d|
          { Time.zone.at(d["startTimeInSeconds"]) => d["steps"] }
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
      # Remove nil values from latest_dates
      latest_dates = latest_dates.compact
      
      # Get the minimum date from the latest dates
      @timestamp = latest_dates.min
      # Round down timestamp to nearest hour
      @timestamp = @timestamp.change(min: 0, sec: 0)
      @timestamp_ago = @timestamp - 1.day

      # Remove chart_data that is not within the range of the minimum date and the date given
      @heart_rate_chart_data = @heart_rate_chart_data.select { 
        |k,v| k >= @timestamp_ago && k <= @timestamp }
      @stress_chart_data = @stress_chart_data.select {
        |k,v| k >= @timestamp_ago && k <= @timestamp }
      @epoch_chart_data = @epoch_chart_data.select {
        |k,v| k >= @timestamp_ago && k <= @timestamp }

      content = render_to_string template: 'users/show.xlsx.axlsx'
      File.open("#{Rails.root}/fastapi/app/tmp2.xlsx", "w+b") { |f| f.puts content }

      # Send to FastAPI for processing
      url = "http://#{ENV['FASTAPI_URL']}:3002/process_api"
      begin
        result = Curl.get(url) do |http|
          http.timeout = 20
          http.ssl_verify_peer = false
        end

        if result.response_code == 200
          @result = JSON.parse(result.body, symbolize_names: false)
          @forecasts = @result["forecasts"].map { |k, v|
            { Time.zone.parse(k) => v.round(4) }
          }.reduce Hash.new, :merge
          
          # Select only the forecasts that are after the timestamp's hour AND only hourly forecasts
          @forecasts = @forecasts.select { |k,v| 
            k.hour > @original_timestamp.hour && k.min == 0 && k.sec == 0
          }
          # Sort the forecasts by date in ascending order
          @forecasts = @forecasts.sort_by { |k,v| k.hour }.to_h
          @forecasts = @forecasts.map { |k,v|
            { k.strftime("%l:%M%p") => v }
          }.reduce Hash.new, :merge
        else
          @forecasts = {}
        end
      end

      json_response message: 'ok', 
        input: {
          id: params[:id],
          timestamp: @original_timestamp
        },
        forecasts: @forecasts
        # result: @result
        # data: {
        #   latest_dates: latest_dates
        # }, 
    end
  end
end
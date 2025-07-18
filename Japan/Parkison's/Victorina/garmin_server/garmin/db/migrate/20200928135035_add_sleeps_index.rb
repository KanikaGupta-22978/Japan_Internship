class AddSleepsIndex < ActiveRecord::Migration[5.2]
	def up
		add_index :sleeps, :data , using: :gin
		execute <<-SQL
			CREATE INDEX index_sleeps_on_data_calendar_date ON sleeps ((data->>'calendarDate'))
		SQL
	end

  def down
  	execute <<-SQL
			DROP INDEX index_sleeps_on_data_calendar_date
		SQL
  	remove_index :sleeps, :data
  end
end

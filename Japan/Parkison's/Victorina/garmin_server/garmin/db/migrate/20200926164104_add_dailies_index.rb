class AddDailiesIndex < ActiveRecord::Migration[5.2]
	def up
		add_index :dailies, :data , using: :gin
		execute <<-SQL
			CREATE INDEX index_dailies_on_data_calendar_date ON dailies ((data->>'calendarDate'))
		SQL
	end

  def down
  	execute <<-SQL
			DROP INDEX index_dailies_on_data_calendar_date
		SQL
  	remove_index :dailies, :data
  end
end

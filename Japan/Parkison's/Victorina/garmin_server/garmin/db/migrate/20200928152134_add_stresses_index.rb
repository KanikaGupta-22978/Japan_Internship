class AddStressesIndex < ActiveRecord::Migration[5.2]
	def up
		add_index :stresses, :data , using: :gin
		execute <<-SQL
			CREATE INDEX index_stresses_on_data_calendar_date ON stresses ((data->>'calendarDate'))
		SQL
	end

  def down
  	execute <<-SQL
			DROP INDEX index_stresses_on_data_calendar_date
		SQL
  	remove_index :stresses, :data
  end
end

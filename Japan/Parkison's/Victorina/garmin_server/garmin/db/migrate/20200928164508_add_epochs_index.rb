class AddEpochsIndex < ActiveRecord::Migration[5.2]
	def change
		add_index :epoches, :data , using: :gin
		add_index :epoches, :calendar_date, using: :btree
	end
end

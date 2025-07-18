class CreatePulseOxes < ActiveRecord::Migration[5.2]
  def change
    create_table :pulse_oxes do |t|
      t.references :user, foreign_key: true
      t.jsonb :data, null: false, default: {}

      t.timestamps
    end
  end
end
